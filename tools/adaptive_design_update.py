"""Create an immutable adaptive-design version from authenticated observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.adaptive_experiment import AdaptiveExperimentError, _validate_design
from tools.experiment_observation import OUTPUT_SCHEMA, canonical_bytes, digest


RECEIPT_SCHEMA = "sim-adaptive-design-update-receipt-v1"


class AdaptiveDesignUpdateError(ValueError):
    """Raised when a design version cannot be updated without widening evidence."""


def _fail(message: str) -> None:
    raise AdaptiveDesignUpdateError(message)


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AdaptiveDesignUpdateError(f"cannot read {field}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{field} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _relative_regular_file(root: Path, path: Path, field: str) -> tuple[str, Path]:
    root = root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AdaptiveDesignUpdateError(f"{field} is missing or outside repository root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} must be a regular file")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            _fail(f"{field} cannot contain a symlink")
    return relative.as_posix(), resolved


def _new_path(root: Path, path: Path, field: str) -> tuple[str, Path]:
    root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise AdaptiveDesignUpdateError(f"{field} is outside repository root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        _fail(f"{field} is not a canonical repository-relative path")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            _fail(f"{field} cannot contain a symlink")
    if os.path.lexists(candidate):
        _fail(f"refusing to replace existing {field}: {candidate}")
    return relative.as_posix(), candidate


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2
    ).encode("ascii") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise AdaptiveDesignUpdateError(f"refusing to replace existing output: {path}") from exc
    path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


def update_adaptive_design(
    design_path: str | Path,
    observation_output_path: str | Path,
    output_path: str | Path,
    receipt_output_path: str | Path,
    *,
    new_id: str,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append only newly authenticated rows and emit a lineage receipt."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    design_relative, design_file = _relative_regular_file(
        root, Path(design_path), "adaptive design"
    )
    observations_relative, observations_file = _relative_regular_file(
        root, Path(observation_output_path), "observation output"
    )
    output_relative, output_file = _new_path(root, Path(output_path), "updated design")
    receipt_relative, receipt_file = _new_path(
        root, Path(receipt_output_path), "update receipt"
    )
    if output_file == receipt_file:
        _fail("updated design and update receipt paths must differ")
    if not isinstance(new_id, str) or not new_id.strip() or new_id != new_id.strip():
        _fail("new_id must be non-empty trimmed text")

    design_raw, design_file_sha = _read_json(design_file, "adaptive design")
    observation_raw, observation_file_sha = _read_json(
        observations_file, "observation output"
    )
    try:
        design = _validate_design(design_raw, root=root)
    except AdaptiveExperimentError as exc:
        raise AdaptiveDesignUpdateError(str(exc)) from exc
    if new_id == design["id"]:
        _fail("new_id must differ from the input design id")

    required = {
        "schema", "canonicalization", "contract", "adaptive_design",
        "executor_manifest", "observations", "evidence", "blocked",
        "scientific_verdict", "sha256",
    }
    if set(observation_raw) != required or observation_raw.get("schema") != OUTPUT_SCHEMA:
        _fail("observation output has an invalid schema or shape")
    expected_self = observation_raw.get("sha256")
    if expected_self != digest({k: v for k, v in observation_raw.items() if k != "sha256"}):
        _fail("observation output self digest is invalid")
    expected_binding = {"path": design_relative, "sha256": design_file_sha}
    if observation_raw.get("adaptive_design") != expected_binding:
        _fail("observation output is not bound to the supplied adaptive design")
    if observation_raw.get("scientific_verdict") is not None:
        _fail("observation output cannot carry a scientific verdict")
    rows = observation_raw.get("observations")
    evidence = observation_raw.get("evidence")
    if not isinstance(rows, list) or not rows:
        _fail("observation output contains no completed observations")
    if not isinstance(evidence, list):
        _fail("observation output evidence must be a list")
    row_ids = [row.get("id") for row in rows if isinstance(row, Mapping)]
    evidence_ids = [row.get("observation_id") for row in evidence if isinstance(row, Mapping)]
    if len(row_ids) != len(rows) or sorted(row_ids) != sorted(evidence_ids):
        _fail("observation rows and evidence bindings are not an exact set")

    old_ids = {row["id"] for row in design["observations"]}
    if old_ids.intersection(row_ids):
        _fail("observation output repeats an existing observation id")
    old_cells = {
        (row["fidelity"], digest(row["parameters"])) for row in design["observations"]
    }
    new_cells = [(row.get("fidelity"), digest(row.get("parameters"))) for row in rows]
    if len(set(new_cells)) != len(new_cells) or old_cells.intersection(new_cells):
        _fail("observation output repeats an existing fidelity/parameter cell")

    updated_raw = {
        **design_raw,
        "id": new_id,
        "observations": [*design_raw["observations"], *rows],
    }
    try:
        _validate_design(updated_raw, root=root)
    except AdaptiveExperimentError as exc:
        raise AdaptiveDesignUpdateError(
            f"updated adaptive design is invalid: {exc}"
        ) from exc

    updated_file_sha = _write_once(output_file, updated_raw)
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "input_design": {"path": design_relative, "sha256": design_file_sha},
        "observation_output": {
            "path": observations_relative,
            "sha256": observation_file_sha,
            "self_sha256": expected_self,
        },
        "updated_design": {"path": output_relative, "sha256": updated_file_sha},
        "previous_id": design["id"],
        "new_id": new_id,
        "previous_observation_count": len(design["observations"]),
        "appended_observation_count": len(rows),
        "updated_observation_count": len(updated_raw["observations"]),
        "blocked_observation_count": len(observation_raw["blocked"]),
        "scientific_verdict": None,
    }
    receipt = {**receipt_body, "sha256": digest(receipt_body)}
    _write_once(receipt_file, receipt)
    return updated_raw, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--new-id", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args(argv)
    try:
        updated, receipt = update_adaptive_design(
            args.design,
            args.observations,
            args.output,
            args.receipt_output,
            new_id=args.new_id,
            repository_root=args.repository_root,
        )
    except AdaptiveDesignUpdateError as exc:
        print(f"adaptive-design-update: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "id": updated["id"],
        "observations": len(updated["observations"]),
        "receipt": receipt["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
