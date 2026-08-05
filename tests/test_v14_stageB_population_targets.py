from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import population_curve_digitization as digitizer
from tools import v14_stageB_population_targets as targets


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, list[dict]]:
    root = tmp_path / "repo"
    protocol_path = root / "specs/protocol.json"
    protocol_sha = _write(protocol_path, {"schema": "measurement-protocol"})
    old_path = root / "specs/old.json"
    old_sha = _write(old_path, {"schema": "old-partition"})
    partition = {
        "schema": targets.PARTITION_SCHEMA,
        "device": "not_applicable_non_executed_protocol",
        "provenance_exempt": "test",
        "status": "preregistered_before_blind_extraction_results_inspected",
        "scientific_verdict": None,
        "optimization_allowed": False,
        "supersedes": {"path": "specs/old.json", "sha256": old_sha},
        "measurement_protocol": {"path": "specs/protocol.json", "sha256": protocol_sha},
        "assignment": {
            "input": "one-based integer suffix of command_NNN after points are sorted by calibrated source x as required by the frozen measurement protocol",
            "held_out": "suffix modulo 5 equals 0",
            "validation": "suffix modulo 5 equals 3",
            "calibration": "all remaining suffixes",
            "uses_x_value": False,
            "uses_y_value": False,
            "uses_uncertainty": False,
            "uses_model_output": False,
            "minimum_commands_per_panel": 5,
            "required_nonempty_partitions_per_panel": list(targets.PARTITIONS),
        },
        "custody": {
            "calibration": "optimizer-visible",
            "validation": "evaluator",
            "held_out": "one-shot",
            "combined_packet_allowed": False,
            "reuse_after_held_out_failure": False,
        },
    }
    partition_path = root / "specs/partition.json"
    _write(partition_path, partition)
    panel = {
        "target_family": "fast_na_activation",
        "asset_id": "asset",
        "panel": "A4",
        "series_identity": "filled",
        "x_quantity": "command_voltage",
        "x_unit": "mV",
        "y_quantity": "normalized_conductance",
        "y_unit": "G/Gmax",
        "sample_size": 12,
        "x_authority_mode": "published_command",
    }
    authority = {"document": {}, "panels": {("asset", "A4"): panel}}
    monkeypatch.setattr(digitizer, "load_protocol", lambda path, root: authority)

    def digitize(record, authority, *, root):
        return {
            "record": {
                "asset": {"asset_id": "asset"},
                "panel": {"id": "A4"},
            },
            "points": [
                {"command_id": f"command_{index:03d}", "authoritative_x": float(index)}
                for index in range(1, 6)
            ],
        }

    def compare(first, second, authority, *, root):
        rows = []
        for index in range(1, 6):
            rows.append(
                {
                    "command_id": f"command_{index:03d}",
                    "accepted": True,
                    "combined_digitized_x": {"median": float(index), "standard_uncertainty": 0.1},
                    "combined_digitized_y": {"median": index / 5, "standard_uncertainty": 0.02},
                    "combined_digitization_uncertainty": {"between_extractor_component": 0.01},
                    "biological_error": {"status": "available", "kind": "standard_error"},
                    "availability": {"first": "available", "second": "available", "unavailable_reason": None},
                }
            )
        result = {
            "schema": digitizer.COMPARISON_SCHEMA,
            "status": "two_extractions_agree",
            "third_extraction_required": False,
            "points": rows,
        }
        result["sha256"] = digitizer.digest(result)
        return result

    monkeypatch.setattr(digitizer, "digitize_record", digitize)
    monkeypatch.setattr(digitizer, "compare_blind_extractions", compare)
    groups = []
    bindings = []
    for name in ("a", "b"):
        path = root / f"measurements/{name}.json"
        sha = _write(path, {"extractor": name})
        bindings.append({"path": f"measurements/{name}.json", "sha256": sha})
    groups.append({"records": bindings})
    return root, protocol_path, partition_path, groups


def test_preregistered_command_assignment_is_value_independent() -> None:
    assert [targets.command_partition(f"command_{index:03d}") for index in range(1, 6)] == [
        "calibration",
        "calibration",
        "validation",
        "calibration",
        "held_out",
    ]
    with pytest.raises(targets.PopulationTargetError, match="command_NNN"):
        targets.command_partition("command_1")


def test_compiler_separates_custody_and_binds_evidence(tmp_path: Path, monkeypatch) -> None:
    root, protocol, partition, groups = _fixture(tmp_path, monkeypatch)
    packets = targets.compile_target_packets(protocol, partition, groups, repository_root=root)

    assert set(packets) == set(targets.PARTITIONS)
    assert [len(packets[name]["targets"]) for name in targets.PARTITIONS] == [3, 1, 1]
    assert packets["calibration"]["proposal_visible"] is True
    assert packets["validation"]["proposal_visible"] is False
    assert packets["held_out"]["proposal_visible"] is False
    assert all(packet["optimization_allowed"] is False for packet in packets.values())
    assert all(
        target["partition"] == name
        for name, packet in packets.items()
        for target in packet["targets"]
    )
    assert packets["calibration"]["evidence"][0]["agreement_schema"] == digitizer.COMPARISON_SCHEMA
    assert "path" not in json.dumps(packets["calibration"]["evidence"])
    assert packets["calibration"]["targets"][0]["x"]["authority"] == "published_command"
    assert packets["calibration"]["targets"][0]["x"]["standard_uncertainty"] == 0.0


def test_compiler_rejects_tamper_missing_panel_and_unresolved_pair(tmp_path: Path, monkeypatch) -> None:
    root, protocol, partition, groups = _fixture(tmp_path, monkeypatch)
    record_path = root / groups[0]["records"][0]["path"]
    record_path.write_text("changed")
    with pytest.raises(targets.PopulationTargetError, match="digest"):
        targets.compile_target_packets(protocol, partition, groups, repository_root=root)

    root, protocol, partition, groups = _fixture(tmp_path / "fresh", monkeypatch)
    monkeypatch.setattr(digitizer, "compare_blind_extractions", lambda *args, **kwargs: {
        "schema": digitizer.COMPARISON_SCHEMA,
        "status": "third_blind_independent_extraction_required",
        "third_extraction_required": True,
        "points": [],
        "sha256": "0" * 64,
    })
    with pytest.raises(targets.PopulationTargetError, match="have not agreed"):
        targets.compile_target_packets(protocol, partition, groups, repository_root=root)


def test_writer_is_create_only_and_index_contains_no_targets(tmp_path: Path, monkeypatch) -> None:
    root, protocol, partition, groups = _fixture(tmp_path, monkeypatch)
    packets = targets.compile_target_packets(protocol, partition, groups, repository_root=root)
    output = root / "runtime/targets"
    index = targets.write_target_packets(output, packets)

    assert index["combined_packet_present"] is False
    assert "targets" not in index
    assert sorted(path.name for path in output.iterdir()) == [
        "calibration.targets.json",
        "held_out.targets.json",
        "index.json",
        "validation.targets.json",
    ]
    with pytest.raises(targets.PopulationTargetError, match="must not already exist"):
        targets.write_target_packets(output, packets)


def test_compiler_rejects_symlinked_evidence(tmp_path: Path, monkeypatch) -> None:
    root, protocol, partition, groups = _fixture(tmp_path, monkeypatch)
    original = root / groups[0]["records"][0]["path"]
    moved = original.with_suffix(".real.json")
    original.rename(moved)
    original.symlink_to(moved)
    groups[0]["records"][0]["sha256"] = hashlib.sha256(moved.read_bytes()).hexdigest()
    with pytest.raises(targets.PopulationTargetError, match="symbolic links"):
        targets.compile_target_packets(protocol, partition, groups, repository_root=root)
