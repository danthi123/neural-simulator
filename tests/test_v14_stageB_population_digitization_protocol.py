import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research/specs/v14_snr_stageB_population_digitization_protocol_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_is_prospective_and_binds_current_authorities():
    protocol = json.loads(PROTOCOL_PATH.read_text())

    assert protocol["status"] == "preregistered_before_numeric_extraction"
    assert protocol["scientific_verdict"] is None
    assert protocol["optimization_allowed"] is False
    for binding in protocol["authorities"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert binding["sha256"] == _sha256(path)


def test_primary_target_assets_have_full_resolution_custody_bindings():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest_binding = protocol["authorities"]["asset_manifest"]
    manifest = json.loads((ROOT / manifest_binding["path"]).read_text())
    required = {panel["asset_id"] for panel in protocol["eligible_panels"]}
    assets = {asset["id"]: asset for asset in manifest["assets"]}

    assert manifest["status"] == "official_full_resolution_assets_hash_bound_numeric_targets_pending"
    for asset_id in required:
        asset = assets[asset_id]
        custody = asset["full_resolution_acquisition"]
        assert custody["tile_count"] > 0
        assert len(custody["manifest_sha256"]) == 64
        assert len(custody["assembled_image_sha256"]) == 64
        assert len(custody["pixel_sha256"]) == 64
        assert len(custody["receipt_file_sha256"]) == 64
        assert len(custody["receipt_self_sha256"]) == 64


def test_protocol_allows_only_population_curve_targets():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    rules = protocol["source_rules"]

    assert rules["representative_single_cell_traces_as_population_targets"] is False
    assert rules["fitted_or_connecting_lines_as_unreported_points"] is False
    assert rules["population_waveform_reconstruction_allowed"] is False
    assert len(protocol["eligible_panels"]) == 7
    assert all(len(panel["panel_bounds_original_pixels"]) == 4 for panel in protocol["eligible_panels"])
    assert all(panel["biological_error_bar"] == "standard_error" for panel in protocol["eligible_panels"])
    kv3_deactivation = next(
        panel for panel in protocol["eligible_panels"] if panel["target_family"] == "kv3_deactivation"
    )
    assert kv3_deactivation["y_scale"] == "log10"
    assert protocol["point_measurement"]["operational_partition_is_not_command_authority"] is True


def test_protocol_separates_measurement_error_from_biological_variation():
    protocol = json.loads(PROTOCOL_PATH.read_text())

    assert protocol["extraction"]["independent_extractors"] == 2
    assert protocol["extraction"]["blind_to_other_extraction"] is True
    assert protocol["digitization_uncertainty"]["monte_carlo_draws"] >= 100_000
    assert protocol["biological_uncertainty"]["stored_separately_from_digitization_uncertainty"] is True
    assert protocol["biological_uncertainty"]["digitization_uncertainty_must_not_be_called_sem"] is True


def test_protocol_fails_closed_on_ambiguous_or_missing_graphics():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    promotion = protocol["promotion_gate"]

    assert protocol["point_measurement"]["point_or_line_interpolation_allowed"] is False
    assert protocol["error_bars"]["hidden_endpoint_inference_allowed"] is False
    assert promotion["held_out_values_visible_to_proposal_generation"] is False
    assert promotion["missing_or_unavailable_commands_are_imputed"] is False
    assert promotion["optimization_remains_blocked_until_verified_target_packet"] is True
