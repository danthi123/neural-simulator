from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tools import pmc_tile_asset
from tools import population_curve_digitization as digitizer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(digitizer.canonical_bytes(value) + b"\n")


def _fixture_root(
    tmp_path: Path, *, log_y: bool = False, digitized_x: bool = False
) -> tuple[Path, Path, dict]:
    root = tmp_path / "repo"
    image_path = root / "runtime/figure.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image_path, format="PNG", optimize=False, compress_level=9)
    pixels = Image.open(image_path).tobytes()
    image_sha, pixel_sha = _sha(image_path), hashlib.sha256(pixels).hexdigest()
    manifest_url = "https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?manifest=1&p=PMC3&id=example.jpg&w=1200&h=900"
    manifest_sha = "1" * 64
    tile_url = "https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?p=PMC3&id=100&s=28&r=1&c=1"
    receipt = {
        "schema": pmc_tile_asset.SCHEMA,
        "scientific_verdict": None,
        "status": "acquired",
        "assembled_image": {"path": "runtime/figure.png", "sha256": image_sha, "pixel_sha256": pixel_sha, "width": 100, "height": 100, "mode": "RGB", "byte_count": image_path.stat().st_size, "content_type": "image/png", "format": "PNG"},
        "manifest": {"url": manifest_url, "sha256": manifest_sha, "text_sha256": "2" * 64, "byte_count": 100, "content_type": "text/plain"},
        "grid": {"column_count": 1, "coordinate_base": 1, "row_count": 1, "tile_count": 1},
        "source": {"image_name": "example.jpg", "project": "PMC3", "satellite": "28"},
        "view": {"sId": "100", "sName": "100%", "fScale": 1, "W": 100, "H": 100, "w": 100, "h": 100},
        "tiles": [{"byte_count": image_path.stat().st_size, "column": 1, "content_type": "image/png", "format": "PNG", "height": 100, "row": 1, "sha256": image_sha, "url": tile_url, "width": 100}],
    }
    receipt["sha256"] = pmc_tile_asset._self_digest(receipt)
    receipt_path = root / "runtime/figure.receipt.json"
    _write_json(receipt_path, receipt)
    # The receipt verifier also authenticates its listed image path.
    manifest = {"schema": "v14-snr-stageB-primary-figure-asset-manifest-v1", "scientific_verdict": None, "assets": [{"id": "asset", "tile_manifest_url": manifest_url, "tile_view_id": "100", "tile_satellite": "28", "full_resolution_pixels": [100, 100], "full_resolution_acquisition": {"local_image_path": "runtime/figure.png", "local_receipt_path": "runtime/figure.receipt.json", "manifest_sha256": manifest_sha, "tile_count": 1, "assembled_image_sha256": image_sha, "pixel_sha256": pixel_sha, "receipt_file_sha256": _sha(receipt_path), "receipt_self_sha256": receipt["sha256"]}}]}
    manifest_path = root / "specs/assets.json"
    _write_json(manifest_path, manifest)
    partition_path = root / "specs/partition.json"
    _write_json(partition_path, {"partition": "placeholder"})
    protocol = {"schema": "v14-snr-stageB-population-digitization-protocol-v1", "status": "preregistered_before_numeric_extraction", "scientific_verdict": None, "optimization_allowed": False, "authorities": {"asset_manifest": {"path": "specs/assets.json", "sha256": _sha(manifest_path)}, "partition": {"path": "specs/partition.json", "sha256": _sha(partition_path)}}, "eligible_panels": [{"asset_id": "asset", "panel": "P", "panel_bounds_original_pixels": [0, 0, 100, 100], "series_identity": "filled", "x_scale": "linear", "y_scale": "log10" if log_y else "linear", "x_authority_mode": "digitized_source_marker" if digitized_x else "published_command", "authoritative_x_values": None if digitized_x else [5], "biological_error_bar": "standard_error"}], "axis_calibration": {"tick_pixel_region_minimum_half_width": 0.5, "tick_pixel_region_maximum_half_width": 8.0, "minimum_calibrated_axis_span_fraction": 0.75}, "point_measurement": {"marker_center_minimum_half_width_pixels": 0.5, "marker_center_maximum_half_width_pixels": 12.0}, "error_bars": {"endpoint_center_maximum_half_width_pixels": 12.0}, "digitization_uncertainty": {"monte_carlo_draws": 1000}, "agreement_and_adjudication": {"threshold": 2.0}}
    protocol_path = root / "specs/protocol.json"
    _write_json(protocol_path, protocol)
    return root, protocol_path, protocol


def _region(x: float, y: float, width: float = 1) -> dict:
    return {"x_min": x - width, "x_max": x + width, "y_min": y - width, "y_max": y + width}


def _record(protocol_path: Path, protocol: dict, *, extractor: str = "extractor_a", x: float = 50, y: float = 50, log_y: bool = False) -> dict:
    root = protocol_path.parents[1]
    image_path = root / "runtime/figure.png"
    receipt_path = root / "runtime/figure.receipt.json"
    y_ticks = [1, 10, 100] if log_y else [0, 0.5, 1]
    panel = protocol["eligible_panels"][0]
    annotation_path = root / f"runtime/{extractor}.annotations.json"
    if not annotation_path.exists():
        _write_json(annotation_path, {"extractor": extractor, "x": x, "y": y})
    return {"schema": digitizer.RECORD_SCHEMA, "record_id": f"record_{extractor}", "extractor_id": extractor, "protocol": {"path": "specs/protocol.json", "sha256": _sha(protocol_path)}, "provenance": {"workflow_id": f"workflow_{extractor}", "blind_to_other_extractions": True, "source_pixels_resampled": False, "tool": {"name": "test_annotation", "version": "1"}, "annotation_artifact": {"path": str(annotation_path.relative_to(root)), "sha256": _sha(annotation_path)}}, "asset": {"asset_id": "asset", "receipt": {"path": "runtime/figure.receipt.json", "file_sha256": _sha(receipt_path), "self_sha256": json.loads(receipt_path.read_text())["sha256"]}, "image": {"path": "runtime/figure.png", "sha256": _sha(image_path), "pixel_sha256": hashlib.sha256(Image.open(image_path).tobytes()).hexdigest()}}, "panel": {"id": "P", "series_id": "filled", "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}}, "status": "available", "unavailable_reason": None, "measurement": {"x_axis": {"scale": "linear", "all_legible_major_ticks_included": True, "outermost_ticks_included": True, "axis_spine_region": {"x_min": 10, "x_max": 90, "y_min": 89, "y_max": 91}, "anchors": [{"pixel_region": _region(10, 90), "value": 0}, {"pixel_region": _region(50, 90), "value": 5}, {"pixel_region": _region(90, 90), "value": 10}]}, "y_axis": {"scale": "log10" if log_y else "linear", "all_legible_major_ticks_included": True, "outermost_ticks_included": True, "axis_spine_region": {"x_min": 9, "x_max": 11, "y_min": 10, "y_max": 90}, "anchors": [{"pixel_region": _region(10, 90), "value": y_ticks[0]}, {"pixel_region": _region(10, 50), "value": y_ticks[1]}, {"pixel_region": _region(10, 10), "value": y_ticks[2]}]}, "points": [{"command_id": "command_001", "authoritative_x": None if panel["x_authority_mode"] == "digitized_source_marker" else 5, "status": "available", "unavailable_reason": None, "marker_center_region": _region(x, y), "occlusion": {"partial": False, "opposing_marker_edges_visible": True, "bounded_without_curve_interpolation": True, "unique_series_identity": True}, "biological_error": {"status": "available", "kind": "standard_error", "lower_endpoint_region": _region(x, y + 5), "upper_endpoint_region": _region(x, y - 5)}}]}}


def test_digitizes_bounded_pixels_deterministically_and_separates_biological_error(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    record = _record(protocol_path, protocol)
    first = digitizer.digitize_record(record, authority, root=root)
    second = digitizer.digitize_record(record, authority, root=root)
    assert first == second
    point = first["points"][0]
    assert point["digitized_y"]["median"] == pytest.approx(0.5, abs=0.03)
    assert point["biological_error"]["kind"] == "standard_error"
    assert "lower_endpoint_digitization" in point["biological_error"]
    assert point["digitization_uncertainty"]["between_extractor_component"] is None
    assert first["scientific_verdict"] is None
    assert first["optimization_command"] is None


def test_log10_axis_uses_logarithmic_calibration(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path, log_y=True)
    output = digitizer.digitize_record(_record(protocol_path, protocol, log_y=True), digitizer.load_protocol(protocol_path, root=root), root=root)
    assert output["points"][0]["digitized_y"]["median"] == pytest.approx(10, rel=0.06)


def test_tampered_image_or_protocol_is_rejected(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    record = _record(protocol_path, protocol)
    (root / "runtime/figure.png").write_bytes(b"changed")
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="digest"):
        digitizer.digitize_record(record, authority, root=root)


def test_ambiguous_or_unbounded_measurements_fail_closed_and_unavailable_records_validate(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    record = _record(protocol_path, protocol)
    record["measurement"]["points"][0]["marker_center_region"] = _region(50, 50, 0.25)
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="narrower"):
        digitizer.digitize_record(record, authority, root=root)
    unavailable = _record(protocol_path, protocol)
    unavailable["status"] = "unavailable"
    unavailable["unavailable_reason"] = "marker_absent_only_line_visible"
    unavailable["measurement"] = None
    output = digitizer.digitize_record(unavailable, authority, root=root)
    assert output["points"] == []


def test_blind_disagreement_requests_a_third_extraction(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    comparison = digitizer.compare_blind_extractions(_record(protocol_path, protocol, extractor="extractor_a", y=20), _record(protocol_path, protocol, extractor="extractor_b", y=80), authority, root=root)
    assert comparison["third_extraction_required"] is True
    assert comparison["status"] == "third_blind_independent_extraction_required"
    assert comparison["scientific_verdict"] is None
    assert comparison["points"][0]["combined_digitization_uncertainty"]["between_extractor_component"] > 0


def test_source_marker_x_has_no_invented_authority_and_is_compared_blindly(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path, digitized_x=True)
    authority = digitizer.load_protocol(protocol_path, root=root)
    left = _record(protocol_path, protocol, extractor="extractor_a", x=20)
    right = _record(protocol_path, protocol, extractor="extractor_b", x=80)

    single = digitizer.digitize_record(left, authority, root=root)
    assert single["points"][0]["authoritative_x"] is None
    comparison = digitizer.compare_blind_extractions(left, right, authority, root=root)
    point = comparison["points"][0]
    assert "x" in point["compared_channels"]
    assert point["normalized_disagreement"]["x"] > point["threshold"]
    assert comparison["third_extraction_required"] is True


def test_third_extraction_resolves_only_one_original_pair(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    first = _record(protocol_path, protocol, extractor="extractor_a", y=20)
    second = _record(protocol_path, protocol, extractor="extractor_b", y=80)
    third = _record(protocol_path, protocol, extractor="extractor_c", y=21)

    result = digitizer.adjudicate_three_extractions(
        first, second, third, authority, root=root
    )

    assert result["status"] == "three_extractions_resolved"
    assert result["points"][0]["resolved_pair"] == "first_third"
    assert len(result["records"]) == 3
    assert result["optimization_allowed"] is False
    assert result["promotion_status"] == "measurement_only"


def test_third_extraction_resolves_biological_error_availability_disagreement(
    tmp_path: Path,
) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    first = _record(protocol_path, protocol, extractor="extractor_a", y=50)
    second = _record(protocol_path, protocol, extractor="extractor_b", y=50)
    third = _record(protocol_path, protocol, extractor="extractor_c", y=50)
    second["measurement"]["points"][0]["biological_error"] = {
        "status": "unavailable"
    }

    comparison = digitizer.compare_blind_extractions(
        first, second, authority, root=root
    )
    point = comparison["points"][0]
    assert point["accepted"] is False
    assert point["biological_error"]["status"] == "extractor_disagreement"
    assert point["biological_error_agreement"]["accepted"] is False
    assert comparison["third_extraction_required"] is True

    result = digitizer.adjudicate_three_extractions(
        first, second, third, authority, root=root
    )
    assert result["status"] == "three_extractions_resolved"
    assert result["points"][0]["resolved_pair"] == "first_third"
    assert result["points"][0]["measurement"]["biological_error"]["status"] == "available"


def test_different_digitized_marker_counts_require_more_blind_evidence(
    tmp_path: Path,
) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path, digitized_x=True)
    authority = digitizer.load_protocol(protocol_path, root=root)
    first = _record(protocol_path, protocol, extractor="extractor_a", x=50)
    second = _record(protocol_path, protocol, extractor="extractor_b", x=50)
    extra = json.loads(json.dumps(second["measurement"]["points"][0]))
    extra["command_id"] = "command_002"
    extra["marker_center_region"] = _region(70, 50)
    extra["biological_error"]["lower_endpoint_region"] = _region(70, 55)
    extra["biological_error"]["upper_endpoint_region"] = _region(70, 45)
    second["measurement"]["points"].append(extra)

    comparison = digitizer.compare_blind_extractions(
        first, second, authority, root=root
    )
    assert comparison["third_extraction_required"] is True
    assert comparison["points"] == []
    assert comparison["command_set_agreement"] == {
        "accepted": False,
        "first": ["command_001"],
        "second": ["command_001", "command_002"],
    }


def test_exact_panel_bounds_command_set_and_error_kind_are_enforced(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)

    wrong_bounds = _record(protocol_path, protocol)
    wrong_bounds["panel"]["bounds"]["x_max"] = 99
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="panel bounds differ"):
        digitizer.digitize_record(wrong_bounds, authority, root=root)

    wrong_command = _record(protocol_path, protocol)
    wrong_command["measurement"]["points"][0]["authoritative_x"] = 6
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="not a published command"):
        digitizer.digitize_record(wrong_command, authority, root=root)

    wrong_error = _record(protocol_path, protocol)
    wrong_error["measurement"]["points"][0]["biological_error"]["kind"] = "standard_deviation"
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="source protocol"):
        digitizer.digitize_record(wrong_error, authority, root=root)


def test_published_command_cell_can_remain_explicitly_unavailable(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    left = _record(protocol_path, protocol, extractor="extractor_a")
    right = _record(protocol_path, protocol, extractor="extractor_b")
    for record in (left, right):
        point = record["measurement"]["points"][0]
        point.update(
            {
                "status": "unavailable",
                "unavailable_reason": "occlusion_prevents_bounded_marker_center",
                "marker_center_region": None,
                "occlusion": None,
                "biological_error": None,
            }
        )

    output = digitizer.digitize_record(left, authority, root=root)
    assert output["points"][0]["status"] == "unavailable"
    comparison = digitizer.compare_blind_extractions(left, right, authority, root=root)
    assert comparison["points"][0]["accepted"] is True
    assert comparison["points"][0]["availability"]["unavailable_reason"] == "occlusion_prevents_bounded_marker_center"


def _whole_panel_unavailable(record: dict, reason: str) -> dict:
    record["status"] = "unavailable"
    record["unavailable_reason"] = reason
    record["measurement"] = None
    return record


def test_blind_whole_panel_unavailability_is_compared_explicitly(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    first = _whole_panel_unavailable(
        _record(protocol_path, protocol, extractor="extractor_a"),
        "occlusion_prevents_bounded_marker_center",
    )
    second = _whole_panel_unavailable(
        _record(protocol_path, protocol, extractor="extractor_b"),
        "occlusion_prevents_bounded_marker_center",
    )
    comparison = digitizer.compare_blind_extractions(first, second, authority, root=root)
    assert comparison["status"] == "two_extractions_agree_panel_unavailable"
    assert comparison["third_extraction_required"] is False
    assert comparison["points"] == []
    assert comparison["panel_availability"]["resolved_unavailable_reason"] == "occlusion_prevents_bounded_marker_center"


def test_third_extraction_can_resolve_panel_availability_either_way(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    available = _record(protocol_path, protocol, extractor="extractor_a", y=50)
    unavailable = _whole_panel_unavailable(
        _record(protocol_path, protocol, extractor="extractor_b"),
        "overlap_prevents_unique_series_assignment",
    )
    third_available = _record(protocol_path, protocol, extractor="extractor_c", y=51)
    numeric = digitizer.adjudicate_three_extractions(
        available, unavailable, third_available, authority, root=root
    )
    assert numeric["status"] == "three_extractions_resolved"
    assert numeric["panel_resolution"]["resolved_pair"] == "first_third"
    assert len(numeric["points"]) == 1

    third_unavailable = _whole_panel_unavailable(
        _record(protocol_path, protocol, extractor="extractor_d"),
        "overlap_prevents_unique_series_assignment",
    )
    withheld = digitizer.adjudicate_three_extractions(
        available, unavailable, third_unavailable, authority, root=root
    )
    assert withheld["status"] == "three_extractions_resolved"
    assert withheld["panel_resolution"]["resolved_pair"] == "second_third"
    assert withheld["points"] == []
    assert withheld["panel_resolution"]["selected_panel_availability"]["resolved_unavailable_reason"] == "overlap_prevents_unique_series_assignment"


def test_four_way_uses_three_of_four_available_panel_vote(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _record(protocol_path, protocol, extractor=extractor, y=50)
        for extractor in ("extractor_a", "extractor_b", "extractor_c")
    ]
    records.append(
        _whole_panel_unavailable(
            _record(protocol_path, protocol, extractor="extractor_d"),
            "occlusion_prevents_bounded_marker_center",
        )
    )

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    assert result["schema"] == digitizer.FOUR_WAY_ADJUDICATION_SCHEMA
    assert result["status"] == "four_extractions_resolved"
    assert result["unresolved"] is False
    assert result["panel_resolution"]["selected_status"] == "available"
    assert result["points"][0]["selected_clique"] == [
        "extractor_a",
        "extractor_b",
        "extractor_c",
    ]


def test_four_way_retains_mixed_majority_panel_unavailability_reasons(
    tmp_path: Path,
) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _whole_panel_unavailable(
            _record(protocol_path, protocol, extractor="extractor_a"),
            "occlusion_prevents_bounded_marker_center",
        ),
        _whole_panel_unavailable(
            _record(protocol_path, protocol, extractor="extractor_b"),
            "overlap_prevents_unique_series_assignment",
        ),
        _whole_panel_unavailable(
            _record(protocol_path, protocol, extractor="extractor_c"),
            "occlusion_prevents_bounded_marker_center",
        ),
        _record(protocol_path, protocol, extractor="extractor_d"),
    ]

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    assert result["status"] == "four_extractions_resolved"
    assert result["panel_resolution"]["status"] == "majority_unavailable_mixed_reasons"
    assert result["panel_resolution"]["unavailable_reasons"] == [
        "occlusion_prevents_bounded_marker_center",
        "overlap_prevents_unique_series_assignment",
    ]
    assert result["panel_status_vote"]["unavailable_votes"] == [
        {
            "extractor_id": "extractor_a",
            "unavailable_reason": "occlusion_prevents_bounded_marker_center",
        },
        {
            "extractor_id": "extractor_b",
            "unavailable_reason": "overlap_prevents_unique_series_assignment",
        },
        {
            "extractor_id": "extractor_c",
            "unavailable_reason": "occlusion_prevents_bounded_marker_center",
        },
    ]


def _four_way_graph_comparison(edges: dict[tuple[str, str], bool], components: dict[tuple[str, str], float]):
    def compare(first, second, authority, *, root):
        pair = tuple(sorted((first["extractor_id"], second["extractor_id"])))
        accepted = edges[pair]
        row = {
            "command_id": "command_001",
            "accepted": accepted,
            "combined_digitized_x": {"median": 5.0, "standard_uncertainty": 0.1},
            "combined_digitized_y": {"median": 0.5, "standard_uncertainty": 0.02},
            "combined_digitization_uncertainty": {
                "between_extractor_component": components[pair]
            },
            "biological_error": {"status": "available", "kind": "standard_error"},
            "availability": {
                "first": "available",
                "second": "available",
                "unavailable_reason": None,
            },
        }
        result = {
            "schema": digitizer.COMPARISON_SCHEMA,
            "points": [row],
            "sha256": "0" * 64,
        }
        return result

    return compare


def test_four_way_selects_unique_three_clique_and_maximum_between_component(
    tmp_path: Path, monkeypatch
) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _record(protocol_path, protocol, extractor=extractor)
        for extractor in ("extractor_a", "extractor_b", "extractor_c", "extractor_d")
    ]
    edges = {
        ("extractor_a", "extractor_b"): True,
        ("extractor_a", "extractor_c"): True,
        ("extractor_a", "extractor_d"): False,
        ("extractor_b", "extractor_c"): True,
        ("extractor_b", "extractor_d"): False,
        ("extractor_c", "extractor_d"): False,
    }
    components = {pair: index / 10 for index, pair in enumerate(sorted(edges), start=1)}
    monkeypatch.setattr(
        digitizer,
        "compare_blind_extractions",
        _four_way_graph_comparison(edges, components),
    )

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    point = result["points"][0]
    assert result["status"] == "four_extractions_resolved"
    assert point["selected_clique"] == ["extractor_a", "extractor_b", "extractor_c"]
    assert point["reported_pair"] == ["extractor_a", "extractor_b"]
    assert point["measurement"]["combined_digitization_uncertainty"][
        "between_extractor_component"
    ] == pytest.approx(max(components[pair] for pair in edges if "extractor_d" not in pair))


def test_four_way_selects_all_four_clique(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _record(protocol_path, protocol, extractor=extractor, y=50)
        for extractor in ("extractor_a", "extractor_b", "extractor_c", "extractor_d")
    ]

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    assert result["status"] == "four_extractions_resolved"
    assert result["points"][0]["selected_clique"] == [
        "extractor_a",
        "extractor_b",
        "extractor_c",
        "extractor_d",
    ]
    assert result["points"][0]["reported_pair"] == ["extractor_a", "extractor_b"]


def test_four_way_bridge_without_three_clique_is_unresolved(tmp_path: Path, monkeypatch) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _record(protocol_path, protocol, extractor=extractor)
        for extractor in ("extractor_a", "extractor_b", "extractor_c", "extractor_d")
    ]
    edges = {
        ("extractor_a", "extractor_b"): True,
        ("extractor_a", "extractor_c"): False,
        ("extractor_a", "extractor_d"): False,
        ("extractor_b", "extractor_c"): True,
        ("extractor_b", "extractor_d"): False,
        ("extractor_c", "extractor_d"): True,
    }
    monkeypatch.setattr(
        digitizer,
        "compare_blind_extractions",
        _four_way_graph_comparison(edges, {pair: 0.1 for pair in edges}),
    )

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    assert result["status"] == "four_extractions_unresolved"
    assert result["points"][0]["status"] == "unresolved"
    assert result["points"][0]["maximum_cliques"] == []


def test_four_way_tied_maximum_cliques_are_unresolved(tmp_path: Path, monkeypatch) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _record(protocol_path, protocol, extractor=extractor)
        for extractor in ("extractor_a", "extractor_b", "extractor_c", "extractor_d")
    ]
    edges = {
        ("extractor_a", "extractor_b"): True,
        ("extractor_a", "extractor_c"): True,
        ("extractor_a", "extractor_d"): True,
        ("extractor_b", "extractor_c"): True,
        ("extractor_b", "extractor_d"): True,
        ("extractor_c", "extractor_d"): False,
    }
    monkeypatch.setattr(
        digitizer,
        "compare_blind_extractions",
        _four_way_graph_comparison(edges, {pair: 0.1 for pair in edges}),
    )

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    assert result["status"] == "four_extractions_unresolved"
    assert result["points"][0]["maximum_cliques"] == [
        ["extractor_a", "extractor_b", "extractor_c"],
        ["extractor_a", "extractor_b", "extractor_d"],
    ]


def test_four_way_uses_exact_digitized_x_command_set_majority(tmp_path: Path) -> None:
    root, protocol_path, protocol = _fixture_root(tmp_path, digitized_x=True)
    authority = digitizer.load_protocol(protocol_path, root=root)
    records = [
        _record(protocol_path, protocol, extractor=extractor, x=50, y=50)
        for extractor in ("extractor_a", "extractor_b", "extractor_c", "extractor_d")
    ]
    extra = json.loads(json.dumps(records[-1]["measurement"]["points"][0]))
    extra["command_id"] = "command_002"
    extra["marker_center_region"] = _region(70, 50)
    extra["biological_error"]["lower_endpoint_region"] = _region(70, 55)
    extra["biological_error"]["upper_endpoint_region"] = _region(70, 45)
    records[-1]["measurement"]["points"].append(extra)

    result = digitizer.adjudicate_four_extractions(*records, authority, root=root)

    assert result["status"] == "four_extractions_resolved"
    assert result["command_set_resolution"]["command_ids"] == ["command_001"]
    assert result["command_set_resolution"]["supporting_extractor_ids"] == [
        "extractor_a",
        "extractor_b",
        "extractor_c",
    ]


def test_create_only_output_is_canonical_and_refuses_replacement(tmp_path: Path) -> None:
    path = tmp_path / "output.json"
    value = {"schema": "test", "scientific_verdict": None, "optimization_command": None}
    digitizer.create_output(path, value)
    assert path.read_bytes() == digitizer.canonical_bytes(value) + b"\n"
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="overwrite"):
        digitizer.create_output(path, value)


def test_manual_measurement_provenance_is_bound_and_create_only(tmp_path: Path) -> None:
    artifact = tmp_path / "annotation.json"
    artifact.write_bytes(b"{}\n")
    sidecar = digitizer.create_provenance_sidecar(
        artifact, role="manual_native_pixel_annotation"
    )
    document = json.loads(sidecar.read_bytes())
    assert document["artifact"]["sha256"] == _sha(artifact)
    assert document["device"] == "human_visual_native_pixels"
    assert document["sim_backend"] == "not_applicable_manual_annotation"
    assert document["sha256"] == digitizer.digest(
        {key: value for key, value in document.items() if key != "sha256"}
    )
    with pytest.raises(digitizer.PopulationCurveDigitizationError, match="overwrite"):
        digitizer.create_provenance_sidecar(
            artifact, role="manual_native_pixel_annotation"
        )
