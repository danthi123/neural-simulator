#!/usr/bin/env python3
"""Seal reconciled population measurements into custody-separated target packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import population_curve_digitization as digitizer


ROOT = Path(__file__).resolve().parents[1]
PARTITION_SCHEMA = "v14-snr-stageB-population-target-partition-v2"
PACKET_SCHEMA = "v14-snr-stageB-population-target-packet-v1"
INDEX_SCHEMA = "v14-snr-stageB-population-target-packet-index-v1"
GROUP_MANIFEST_SCHEMA = "v14-snr-stageB-population-extraction-group-manifest-v1"
PARTITIONS = ("calibration", "validation", "held_out")
_COMMAND = re.compile(r"command_([0-9]{3})\Z")


class PopulationTargetError(ValueError):
    """Raised when source evidence cannot be sealed without leakage or ambiguity."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise PopulationTargetError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any, context: str) -> str:
    _fail(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{context} must be a lowercase SHA-256 digest",
    )
    return value


def _repo_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    _fail(isinstance(value, str) and value == value.strip() and value, f"{context} path is invalid")
    supplied = Path(value).expanduser()
    _fail(".." not in supplied.parts, f"{context} path is not canonical")
    lexical = supplied if supplied.is_absolute() else root / supplied
    lexical = Path(os.path.abspath(lexical))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise PopulationTargetError(f"{context} must be inside repository_root") from exc
    probe = lexical
    while probe != root:
        _fail(not probe.is_symlink(), f"{context} must not use symbolic links")
        probe = probe.parent
    path = lexical.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise PopulationTargetError(f"{context} must be inside repository_root") from exc
    _fail(path.is_file(), f"{context} must be a regular file")
    pure = PurePosixPath(relative.as_posix())
    _fail(all(part not in {"", ".", ".."} for part in pure.parts), f"{context} path is not canonical")
    return pure.as_posix(), path


def _bound_json(root: Path, binding: Any, context: str) -> tuple[dict[str, str], dict[str, Any]]:
    _fail(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"}, f"{context} binding is invalid")
    relative, path = _repo_file(root, binding["path"], context)
    expected = _sha(binding["sha256"], f"{context} sha256")
    _fail(_file_digest(path) == expected, f"{context} digest does not match")
    try:
        document = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise PopulationTargetError(f"{context} is not valid JSON") from exc
    _fail(isinstance(document, dict), f"{context} must contain an object")
    return {"path": relative, "sha256": expected}, document


def _load_partition(root: Path, path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    relative, absolute = _repo_file(root, str(path), "partition protocol")
    try:
        document = json.loads(absolute.read_bytes())
    except json.JSONDecodeError as exc:
        raise PopulationTargetError("partition protocol is not valid JSON") from exc
    _fail(document.get("schema") == PARTITION_SCHEMA, "partition protocol schema is invalid")
    _fail(
        document.get("status") == "preregistered_before_blind_extraction_results_inspected"
        and document.get("scientific_verdict") is None
        and document.get("optimization_allowed") is False,
        "partition protocol is not a sealed non-result protocol",
    )
    expected_assignment = {
        "input": "one-based integer suffix of command_NNN after points are sorted by calibrated source x as required by the frozen measurement protocol",
        "held_out": "suffix modulo 5 equals 0",
        "validation": "suffix modulo 5 equals 3",
        "calibration": "all remaining suffixes",
        "uses_x_value": False,
        "uses_y_value": False,
        "uses_uncertainty": False,
        "uses_model_output": False,
        "minimum_commands_per_panel": 5,
        "required_nonempty_partitions_per_panel": list(PARTITIONS),
    }
    _fail(document.get("assignment") == expected_assignment, "partition assignment differs from the preregistration")
    for name in ("measurement_protocol", "supersedes"):
        reference, _ = _bound_json(root, document.get(name), f"partition {name}")
        _fail(reference == document[name], f"partition {name} binding is not canonical")
    custody = document.get("custody")
    _fail(
        isinstance(custody, Mapping)
        and custody.get("combined_packet_allowed") is False
        and custody.get("reuse_after_held_out_failure") is False,
        "partition custody is invalid",
    )
    return {"path": relative, "sha256": _file_digest(absolute)}, document


def command_partition(command_id: str) -> str:
    """Apply the preregistered assignment without reading a measured value."""

    match = _COMMAND.fullmatch(command_id) if isinstance(command_id, str) else None
    _fail(match is not None, "command id must have the command_NNN form")
    index = int(match.group(1))
    _fail(index >= 1, "command suffix must be one-based")
    if index % 5 == 0:
        return "held_out"
    if index % 5 == 3:
        return "validation"
    return "calibration"


def build_extraction_group_manifest(
    protocol_path: str | Path,
    record_paths: Sequence[str | Path],
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Discover complete blind panel groups without manually pairing paths."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    protocol_relative, protocol_absolute = _repo_file(
        root, str(protocol_path), "measurement protocol"
    )
    authority = digitizer.load_protocol(protocol_absolute, root=root)
    protocol_binding = {
        "path": protocol_relative,
        "sha256": _file_digest(protocol_absolute),
    }
    _fail(
        isinstance(record_paths, Sequence)
        and not isinstance(record_paths, (str, bytes))
        and bool(record_paths),
        "record paths must be a non-empty sequence",
    )
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = {}
    seen_paths: set[str] = set()
    for index, supplied in enumerate(record_paths):
        relative, path = _repo_file(root, str(supplied), f"record {index}")
        _fail(relative not in seen_paths, "record paths must be unique")
        seen_paths.add(relative)
        try:
            record = json.loads(path.read_bytes())
        except json.JSONDecodeError as exc:
            raise PopulationTargetError(f"record {index} is not valid JSON") from exc
        normalized = digitizer.validate_extraction_record(record, authority, root=root)
        key = (normalized["asset"]["asset_id"], normalized["panel"]["id"])
        grouped.setdefault(key, []).append(
            (
                normalized["extractor_id"],
                {"path": relative, "sha256": _file_digest(path)},
            )
        )
    _fail(set(grouped) == set(authority["panels"]), "records do not cover every eligible panel")
    extraction_groups: list[dict[str, Any]] = []
    panel_index: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: item[0])
        extractor_ids = [item[0] for item in rows]
        _fail(
            len(rows) in {2, 3, 4} and len(set(extractor_ids)) == len(rows),
            f"panel {key} must have two, three, or four distinct extractors",
        )
        extraction_groups.append({"records": [item[1] for item in rows]})
        panel_index.append(
            {
                "asset_id": key[0],
                "panel": key[1],
                "extractor_ids": extractor_ids,
            }
        )
    core = {
        "schema": GROUP_MANIFEST_SCHEMA,
        "scientific_verdict": None,
        "optimization_allowed": False,
        "status": "authenticated_blind_extraction_groups",
        "measurement_protocol": protocol_binding,
        "panel_index": panel_index,
        "extraction_groups": extraction_groups,
    }
    core["sha256"] = _digest(core)
    return core


def _measurement_rows(
    result: Mapping[str, Any], count: int
) -> tuple[list[dict[str, Any]], str | None]:
    if count == 2:
        _fail(
            result.get("schema") == digitizer.COMPARISON_SCHEMA
            and result.get("third_extraction_required") is False,
            "two blind extractions have not agreed",
        )
        if result.get("status") == "two_extractions_agree_panel_unavailable":
            reason = result["panel_availability"].get("resolved_unavailable_reason")
            _fail(isinstance(reason, str) and reason, "panel unavailability lacks a reason")
            return [], reason
        _fail(result.get("status") == "two_extractions_agree", "blind agreement status is invalid")
        rows = result.get("points")
    elif count == 3:
        _fail(
            result.get("schema") == digitizer.ADJUDICATION_SCHEMA
            and result.get("status") == "three_extractions_resolved"
            and result.get("unresolved") is False,
            "three-way adjudication remains unresolved",
        )
        panel_resolution = result.get("panel_resolution")
        if (
            isinstance(panel_resolution, Mapping)
            and isinstance(panel_resolution.get("selected_panel_availability"), Mapping)
            and panel_resolution["selected_panel_availability"].get("both_available") is False
        ):
            reason = panel_resolution["selected_panel_availability"].get(
                "resolved_unavailable_reason"
            )
            _fail(isinstance(reason, str) and reason, "adjudicated panel unavailability lacks a reason")
            return [], reason
        rows = []
        for point in result.get("points", []):
            _fail(point.get("status") == "resolved", "adjudication contains an unresolved point")
            rows.append(point["measurement"])
    else:
        _fail(count == 4, "measurement result has an unsupported extractor count")
        _fail(
            result.get("schema") == digitizer.FOUR_WAY_ADJUDICATION_SCHEMA
            and result.get("status") == "four_extractions_resolved"
            and result.get("unresolved") is False,
            "four-way adjudication remains unresolved",
        )
        panel_resolution = result.get("panel_resolution")
        _fail(isinstance(panel_resolution, Mapping), "four-way panel resolution is invalid")
        selected_status = panel_resolution.get("selected_status")
        if selected_status == "unavailable":
            reasons = panel_resolution.get("unavailable_reasons")
            _fail(
                isinstance(reasons, list)
                and reasons
                and all(isinstance(reason, str) and reason for reason in reasons),
                "four-way panel unavailability lacks its preregistered reasons",
            )
            return [], (
                "majority_unavailable_mixed_reasons"
                if len(reasons) > 1
                else reasons[0]
            )
        _fail(selected_status == "available", "four-way panel status is invalid")
        command_resolution = result.get("command_set_resolution")
        _fail(
            isinstance(command_resolution, Mapping)
            and command_resolution.get("status") == "resolved"
            and isinstance(command_resolution.get("command_ids"), list),
            "four-way command-set consensus is invalid",
        )
        rows = []
        for point in result.get("points", []):
            _fail(point.get("status") == "resolved", "four-way adjudication contains an unresolved point")
            rows.append(point["measurement"])
    _fail(isinstance(rows, list) and rows, "measurement result has no points")
    return [dict(row) for row in rows], None


def _target(
    panel_protocol: Mapping[str, Any], reference_point: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    command_id = row.get("command_id")
    partition = command_partition(command_id)
    availability = row.get("availability")
    _fail(isinstance(availability, Mapping), f"{command_id} availability is invalid")
    available = availability.get("first") == availability.get("second") == "available"
    if available:
        _fail(
            row.get("accepted") is True
            and isinstance(row.get("combined_digitized_x"), Mapping)
            and isinstance(row.get("combined_digitized_y"), Mapping),
            f"{command_id} lacks an accepted numeric measurement",
        )
    else:
        _fail(
            availability.get("first") == availability.get("second") == "unavailable"
            and isinstance(availability.get("unavailable_reason"), str),
            f"{command_id} availability disagreement is unresolved",
        )
    if available and panel_protocol["x_authority_mode"] == "published_command":
        authoritative_x = reference_point.get("authoritative_x")
        _fail(isinstance(authoritative_x, (int, float)) and not isinstance(authoritative_x, bool), f"{command_id} lacks its published command")
        x_measurement = {
            "median": float(authoritative_x),
            "standard_uncertainty": 0.0,
            "q025": float(authoritative_x),
            "q975": float(authoritative_x),
            "authority": "published_command",
        }
    else:
        x_measurement = row.get("combined_digitized_x") if available else None
    return {
        "target_id": f"{panel_protocol['target_family']}:{command_id}",
        "target_family": panel_protocol["target_family"],
        "asset_id": panel_protocol["asset_id"],
        "panel": panel_protocol["panel"],
        "series_identity": panel_protocol["series_identity"],
        "command_id": command_id,
        "partition": partition,
        "x_quantity": panel_protocol["x_quantity"],
        "x_unit": panel_protocol["x_unit"],
        "y_quantity": panel_protocol["y_quantity"],
        "y_unit": panel_protocol["y_unit"],
        "sample_size": panel_protocol.get("sample_size"),
        "measurement_limitation": panel_protocol.get("measurement_limitation"),
        "status": "available" if available else "unavailable",
        "unavailable_reason": None if available else availability["unavailable_reason"],
        "x": x_measurement,
        "y": row.get("combined_digitized_y") if available else None,
        "digitization_uncertainty": row.get("combined_digitization_uncertainty") if available else None,
        "biological_error": row.get("biological_error") if available else None,
    }


def compile_target_packets(
    protocol_path: str | Path,
    partition_path: str | Path,
    extraction_groups: Sequence[Mapping[str, Any]],
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Recompute blind agreement and emit three mutually exclusive packets."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    protocol_relative, protocol_absolute = _repo_file(root, str(protocol_path), "measurement protocol")
    authority = digitizer.load_protocol(protocol_absolute, root=root)
    protocol_binding = {"path": protocol_relative, "sha256": _file_digest(protocol_absolute)}
    partition_binding, partition = _load_partition(root, partition_path)
    _fail(partition["measurement_protocol"] == protocol_binding, "partition binds a different measurement protocol")
    _fail(isinstance(extraction_groups, Sequence) and not isinstance(extraction_groups, (str, bytes)), "extraction groups must be a sequence")

    targets: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    unavailable_panels: list[dict[str, Any]] = []
    seen_panels: set[tuple[str, str]] = set()
    for group_index, group in enumerate(extraction_groups):
        _fail(isinstance(group, Mapping) and set(group) == {"records"}, f"extraction group {group_index} is invalid")
        bindings = group["records"]
        _fail(isinstance(bindings, list) and len(bindings) in {2, 3, 4}, f"extraction group {group_index} must contain two, three, or four records")
        loaded: list[dict[str, Any]] = []
        canonical_bindings: list[dict[str, str]] = []
        for record_index, binding in enumerate(bindings):
            canonical, record = _bound_json(root, binding, f"extraction group {group_index} record {record_index}")
            canonical_bindings.append(canonical)
            loaded.append(record)
        digitized_outputs = (
            [digitizer.digitize_record(record, authority, root=root) for record in loaded]
            if len(loaded) == 4
            else [digitizer.digitize_record(loaded[0], authority, root=root)]
        )
        first_output = digitized_outputs[0]
        panel_record = first_output["record"]
        panel_key = (panel_record["asset"]["asset_id"], panel_record["panel"]["id"])
        _fail(panel_key not in seen_panels, f"duplicate extraction group for panel {panel_key}")
        seen_panels.add(panel_key)
        panel_protocol = authority["panels"].get(panel_key)
        _fail(panel_protocol is not None, f"panel {panel_key} is not eligible")
        if len(loaded) == 2:
            result = digitizer.compare_blind_extractions(*loaded, authority, root=root)
        elif len(loaded) == 3:
            result = digitizer.adjudicate_three_extractions(*loaded, authority, root=root)
        else:
            result = digitizer.adjudicate_four_extractions(*loaded, authority, root=root)
        rows, unavailable_reason = _measurement_rows(result, len(loaded))
        if unavailable_reason is not None:
            unavailable_panel = {
                "target_family": panel_protocol["target_family"],
                "asset_id": panel_key[0],
                "panel": panel_key[1],
                "unavailable_reason": unavailable_reason,
            }
            if len(loaded) == 4:
                reasons = result["panel_resolution"]["unavailable_reasons"]
                unavailable_panel["unavailable_reasons"] = list(reasons)
            unavailable_panels.append(unavailable_panel)
            panel_targets = []
        else:
            if len(loaded) == 4:
                command_ids = tuple(result["command_set_resolution"]["command_ids"])
                supporting_outputs = [
                    output
                    for output in digitized_outputs
                    if output["record"]["status"] == "available"
                    and tuple(sorted(point["command_id"] for point in output["points"]))
                    == command_ids
                ]
                _fail(
                    len(supporting_outputs) >= 3,
                    "four-way consensus lacks three authenticated command-set voters",
                )
                reference_output = min(
                    supporting_outputs,
                    key=lambda output: output["record"]["extractor_id"],
                )
            else:
                reference_output = first_output
            reference_points = {
                point["command_id"]: point for point in reference_output.get("points", [])
            }
            _fail(
                set(reference_points) == {row.get("command_id") for row in rows},
                f"panel {panel_key} agreement commands differ from the authenticated extraction",
            )
            panel_targets = [
                _target(panel_protocol, reference_points[row["command_id"]], row)
                for row in rows
            ]
        if panel_targets:
            _fail(len(panel_targets) >= partition["assignment"]["minimum_commands_per_panel"], f"panel {panel_key} has too few commands")
            present = {target["partition"] for target in panel_targets}
            _fail(present == set(PARTITIONS), f"panel {panel_key} does not populate every partition")
        targets.extend(panel_targets)
        evidence.append(
            {
                "asset_id": panel_key[0],
                "panel": panel_key[1],
                "record_file_sha256": [binding["sha256"] for binding in canonical_bindings],
                "agreement_schema": result["schema"],
                "agreement_sha256": result["sha256"],
            }
        )

    expected_panels = set(authority["panels"])
    _fail(seen_panels == expected_panels, "extraction groups do not cover every eligible panel exactly once")
    targets.sort(key=lambda row: (row["target_family"], row["command_id"]))
    evidence.sort(key=lambda row: (row["asset_id"], row["panel"]))
    unavailable_panels.sort(key=lambda row: (row["target_family"], row["panel"]))
    packets: dict[str, dict[str, Any]] = {}
    for name in PARTITIONS:
        core = {
            "schema": PACKET_SCHEMA,
            "scientific_verdict": None,
            "optimization_command": None,
            "optimization_allowed": False,
            "status": "sealed_source_measurements",
            "partition": name,
            "proposal_visible": name == "calibration",
            "measurement_protocol": protocol_binding,
            "partition_protocol": partition_binding,
            "evidence": evidence,
            "unavailable_panels": unavailable_panels,
            "targets": [target for target in targets if target["partition"] == name],
        }
        _fail(core["targets"], f"{name} target packet is empty")
        core["sha256"] = _digest(core)
        packets[name] = core
    return packets


def write_target_packets(output_dir: str | Path, packets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Create custody-separated files and a digest-only index without overwriting."""

    destination = Path(output_dir).expanduser().resolve()
    _fail(not destination.exists() and not destination.is_symlink(), "output directory must not already exist")
    _fail(set(packets) == set(PARTITIONS), "exactly three target packets are required")
    destination.mkdir(parents=True)
    references: dict[str, dict[str, str]] = {}
    try:
        for name in PARTITIONS:
            packet = dict(packets[name])
            _fail(packet.get("partition") == name and packet.get("sha256") == _digest({k: v for k, v in packet.items() if k != "sha256"}), f"{name} packet digest is invalid")
            path = destination / f"{name}.targets.json"
            raw = _canonical(packet) + b"\n"
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            references[name] = {"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
        index = {
            "schema": INDEX_SCHEMA,
            "scientific_verdict": None,
            "optimization_allowed": False,
            "combined_packet_present": False,
            "packets": references,
        }
        index["sha256"] = _digest(index)
        index_path = destination / "index.json"
        with index_path.open("xb") as handle:
            handle.write(_canonical(index) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        for child in destination.iterdir():
            child.unlink()
        destination.rmdir()
        raise
    return index


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--groups", required=True, help="JSON file containing extraction_groups")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    root = Path(args.repository_root).expanduser().resolve(strict=True)
    groups_relative, groups_path = _repo_file(root, args.groups, "extraction groups")
    _, groups_document = _bound_json(
        root,
        {"path": groups_relative, "sha256": _file_digest(groups_path)},
        "extraction groups",
    )
    _fail(
        set(groups_document)
        == {
            "schema",
            "scientific_verdict",
            "optimization_allowed",
            "status",
            "measurement_protocol",
            "panel_index",
            "extraction_groups",
            "sha256",
        }
        and groups_document.get("schema") == GROUP_MANIFEST_SCHEMA
        and groups_document.get("scientific_verdict") is None
        and groups_document.get("optimization_allowed") is False
        and groups_document.get("status") == "authenticated_blind_extraction_groups"
        and groups_document.get("sha256")
        == _digest({key: value for key, value in groups_document.items() if key != "sha256"}),
        "groups document is invalid",
    )
    expected_protocol = {
        "path": _repo_file(root, args.protocol, "measurement protocol")[0],
        "sha256": _file_digest(_repo_file(root, args.protocol, "measurement protocol")[1]),
    }
    _fail(
        groups_document["measurement_protocol"] == expected_protocol,
        "groups document binds a different measurement protocol",
    )
    packets = compile_target_packets(
        args.protocol,
        args.partition,
        groups_document["extraction_groups"],
        repository_root=root,
    )
    write_target_packets(args.output, packets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
