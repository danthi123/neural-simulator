from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = ROOT / "research/specs/v14_snr_stageB_kinetic_identification_partition_v1.json"
ASSET_MANIFEST_PATH = ROOT / "research/specs/v14_snr_stageB_primary_figure_asset_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def test_partition_binds_the_exact_command_authority() -> None:
    partition = _load(PARTITION_PATH)
    binding = partition["command_authority"]
    source = ROOT / binding["path"]

    assert hashlib.sha256(source.read_bytes()).hexdigest() == binding["sha256"]
    assert partition["status"] == "blocked_pending_population_target_digitization_and_scope_correction"
    assert partition["execution_allowed"] is False
    assert partition["scientific_verdict"] is None
    assert "raw population current-versus-time waveforms are unavailable" in (
        partition["source_target_requirements"]["current_availability"]
    )
    assert partition["source_target_requirements"]["synthetic_targets_allowed"] is False
    assert partition["source_target_requirements"]["population_waveform_reconstruction_allowed"] is False


def test_each_partition_is_disjoint_and_exactly_covers_the_source_ladder() -> None:
    partition = _load(PARTITION_PATH)
    commands = _load(ROOT / partition["command_authority"]["path"])["commands"]
    source_fields = {
        "fast_na_activation_test_mV": ("fast_na_activation", "test_mV"),
        "fast_na_inactivation_prepulse_mV": ("fast_na_inactivation", "prepulse_mV"),
        "fast_na_recovery_duration_ms": ("fast_na_recovery", "recovery_duration_ms"),
        "fast_na_deactivation_test_mV": ("fast_na_deactivation", "test_mV"),
        "kv3_activation_test_mV": ("kv3_activation", "test_mV"),
        "kv3_inactivation_prepulse_mV": ("kv3_inactivation", "prepulse_mV"),
        "kv3_deactivation_test_mV": ("kv3_deactivation", "test_mV"),
    }

    assert set(partition["partitions"]) == set(source_fields)
    for name, (assay, field) in source_fields.items():
        split = partition["partitions"][name]
        assert set(split) == {"calibration", "validation", "held_out"}
        cells = [set(split[key]) for key in ("calibration", "validation", "held_out")]
        assert all(cells)
        assert cells[0].isdisjoint(cells[1])
        assert cells[0].isdisjoint(cells[2])
        assert cells[1].isdisjoint(cells[2])
        assert set().union(*cells) == set(commands[assay][field])


def test_diagnostic_anchor_commands_remain_held_out() -> None:
    partitions = _load(PARTITION_PATH)["partitions"]

    assert 0 in partitions["fast_na_activation_test_mV"]["held_out"]
    assert -40 in partitions["fast_na_deactivation_test_mV"]["held_out"]
    assert 40 in partitions["kv3_activation_test_mV"]["held_out"]
    assert -50 in partitions["kv3_deactivation_test_mV"]["held_out"]
    assert _load(PARTITION_PATH)["partition_policy"]["reuse_after_held_out_failure"] is False


def test_primary_asset_manifest_separates_population_curves_from_trace_context() -> None:
    manifest = _load(ASSET_MANIFEST_PATH)
    assets = manifest["assets"]

    assert manifest["scientific_verdict"] is None
    assert manifest["target_policy"]["representative_trace_as_population_mean_allowed"] is False
    assert manifest["target_policy"]["raw_complete_current_waveforms_available"] is False
    assert len({row["id"] for row in assets}) == len(assets) == 7
    assert all(row["url"].startswith("https://cdn.ncbi.nlm.nih.gov/pmc/") for row in assets)
    assert all("manifest=1" in row["tile_manifest_url"] for row in assets)
    assert all(len(row["sha256"]) == 64 for row in assets)
    assert all(row["pixels"][0] > 0 and row["pixels"][1] > 0 for row in assets)
    assert all(row["full_resolution_pixels"][0] >= row["pixels"][0]
               and row["full_resolution_pixels"][1] >= row["pixels"][1] for row in assets)
    assert all(row["tile_view_id"].isdigit() and row["tile_satellite"].isdigit()
               for row in assets)
    assert all("representative" in " ".join(row["content"]) for row in assets)
    assert all("population" in row["allowed_numeric_role"] or "validation" in row["allowed_numeric_role"]
               for row in assets)
