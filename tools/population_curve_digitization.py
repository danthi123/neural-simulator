#!/usr/bin/env python3
"""Validate and propagate bounded Stage B population-curve digitizations.

This is a source-measurement utility, not a model scorer.  It accepts only
explicitly bounded marker and error-bar observations made in original image
pixels, binds them to the acquired PMC asset, and reports digitization error
separately from the paper's biological uncertainty.  It never creates a
scientific verdict, an optimization command, or an inferred graphical value.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from itertools import combinations
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np
import PIL

try:  # Supports both ``python -m tools...`` and direct tool invocation.
    from tools import pmc_tile_asset
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI smoke check.
    import pmc_tile_asset  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
RECORD_SCHEMA = "v14-snr-stageB-population-curve-extraction-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-population-curve-digitization-v1"
COMPARISON_SCHEMA = "v14-snr-stageB-population-curve-agreement-v1"
ADJUDICATION_SCHEMA = "v14-snr-stageB-population-curve-adjudication-v1"
FOUR_WAY_ADJUDICATION_SCHEMA = "v14-snr-stageB-population-curve-four-way-adjudication-v1"
PROVENANCE_SCHEMA = "v14-snr-stageB-manual-measurement-provenance-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_UNAVAILABLE_REASONS = frozenset(
    {
        "asset_digest_or_original_dimensions_mismatch",
        "panel_axis_unit_or_series_identity_ambiguous",
        "fewer_than_three_usable_ticks",
        "calibration_acceptance_failed",
        "marker_absent_only_line_visible",
        "overlap_prevents_unique_series_assignment",
        "occlusion_prevents_bounded_marker_center",
        "digitized_x_conflicts_with_authoritative_command",
        "error_bar_endpoint_not_distinguishable",
        "duplicate_extractions_disagree_unresolved",
    }
)


class PopulationCurveDigitizationError(ValueError):
    """Raised when a record is outside the preregistered measurement protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PopulationCurveDigitizationError(message)


def canonical_bytes(value: Any) -> bytes:
    """Encode a deterministic, finite JSON document."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PopulationCurveDigitizationError("value is not canonical JSON data") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _runtime() -> dict[str, str]:
    return {
        "algorithm": "population_curve_digitization_v1",
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "python": platform.python_version(),
        "quantile_method": "linear",
    }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, context: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{context} is not a SHA-256 digest")
    return value


def _finite(value: Any, context: str) -> float:
    _require(type(value) in {int, float} and math.isfinite(float(value)), f"{context} must be finite")
    return float(value)


def _strict_object(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == keys, f"{context} fields are invalid")
    return value


def _repository_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    _require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value, f"{context} path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and str(relative) == value
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{context} path is not canonical",
    )
    candidate = root.joinpath(*relative.parts)
    _require(not candidate.is_symlink(), f"{context} must not be a symbolic link")
    path = candidate.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PopulationCurveDigitizationError(f"{context} escapes repository") from exc
    _require(path.is_file(), f"{context} is unavailable")
    return value, path


def _load_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PopulationCurveDigitizationError(f"{context} is not valid JSON") from exc
    _require(isinstance(value, Mapping), f"{context} must be an object")
    return value


def _binding(root: Path, value: Any, context: str) -> tuple[dict[str, str], Path]:
    binding = _strict_object(value, {"path", "sha256"}, context)
    relative, path = _repository_file(root, binding["path"], context)
    observed = _file_digest(path)
    _require(observed == _sha256(binding["sha256"], f"{context} digest"), f"{context} digest mismatch")
    return {"path": relative, "sha256": observed}, path


def load_protocol(protocol_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Load a protocol only when its asset-manifest authority is still exact."""
    root = Path(root).resolve()
    protocol = dict(_load_json(Path(protocol_path), "protocol"))
    _require(protocol.get("schema") == "v14-snr-stageB-population-digitization-protocol-v1", "protocol schema is invalid")
    _require(protocol.get("status") == "preregistered_before_numeric_extraction", "protocol is not prospective")
    _require(protocol.get("scientific_verdict") is None, "protocol contains a scientific verdict")
    _require(protocol.get("optimization_allowed") is False, "protocol permits optimization")
    authorities = _strict_object(protocol.get("authorities"), {"asset_manifest", "partition"}, "protocol authorities")
    manifest_binding, manifest_path = _binding(root, authorities["asset_manifest"], "asset manifest")
    _binding(root, authorities["partition"], "partition")
    manifest = _load_json(manifest_path, "asset manifest")
    _require(manifest.get("schema") == "v14-snr-stageB-primary-figure-asset-manifest-v1", "asset manifest schema is invalid")
    _require(manifest.get("scientific_verdict") is None, "asset manifest contains a scientific verdict")
    assets = manifest.get("assets")
    _require(isinstance(assets, list), "asset manifest assets are invalid")
    asset_rows = {row.get("id"): row for row in assets if isinstance(row, Mapping)}
    _require(len(asset_rows) == len(assets), "asset manifest has duplicate or invalid asset ids")
    panels = protocol.get("eligible_panels")
    _require(isinstance(panels, list) and panels, "protocol has no eligible panels")
    panel_rows = {(row.get("asset_id"), row.get("panel")): row for row in panels if isinstance(row, Mapping)}
    _require(len(panel_rows) == len(panels), "protocol has duplicate or invalid eligible panels")
    for (asset_id, panel), row in panel_rows.items():
        _require(asset_id in asset_rows and isinstance(panel, str) and panel, "protocol panel is not asset-bound")
        _require(row.get("x_scale") in {"linear", "log10"} and row.get("y_scale") in {"linear", "log10"}, "protocol scale is invalid")
    return {
        "document": protocol,
        "binding": {"path": str(Path(protocol_path).resolve().relative_to(root)), "sha256": _file_digest(Path(protocol_path))},
        "asset_manifest_binding": manifest_binding,
        "assets": asset_rows,
        "panels": panel_rows,
    }


def _region(value: Any, context: str) -> dict[str, float]:
    row = _strict_object(value, {"x_min", "x_max", "y_min", "y_max"}, context)
    answer = {key: _finite(item, f"{context} {key}") for key, item in row.items()}
    _require(answer["x_min"] <= answer["x_max"] and answer["y_min"] <= answer["y_max"], f"{context} bounds are invalid")
    return answer


def _center(region: Mapping[str, float], axis: str) -> float:
    return (region[f"{axis}_min"] + region[f"{axis}_max"]) / 2.0


def _half_width(region: Mapping[str, float], axis: str) -> float:
    return (region[f"{axis}_max"] - region[f"{axis}_min"]) / 2.0


def _inside_panel(region: Mapping[str, float], bounds: Mapping[str, float], context: str) -> None:
    _require(
        bounds["x_min"] <= region["x_min"] <= region["x_max"] <= bounds["x_max"]
        and bounds["y_min"] <= region["y_min"] <= region["y_max"] <= bounds["y_max"],
        f"{context} is outside panel",
    )


def _validate_axis(
    value: Any,
    axis: str,
    expected_scale: str,
    minimum_half_width: float,
    maximum_half_width: float,
) -> dict[str, Any]:
    row = _strict_object(
        value,
        {"all_legible_major_ticks_included", "anchors", "axis_spine_region", "outermost_ticks_included", "scale"},
        f"{axis} calibration",
    )
    _require(row["scale"] == expected_scale, f"{axis} calibration scale differs from protocol")
    _require(row["all_legible_major_ticks_included"] is True, f"{axis} calibration omits a legible major tick")
    _require(row["outermost_ticks_included"] is True, f"{axis} calibration omits an outermost tick")
    anchors = row["anchors"]
    _require(isinstance(anchors, list) and len(anchors) >= 3, f"{axis} calibration needs three anchors")
    result: list[dict[str, Any]] = []
    seen_values: set[float] = set()
    for index, anchor in enumerate(anchors):
        item = _strict_object(anchor, {"pixel_region", "value"}, f"{axis} anchor {index}")
        region = _region(item["pixel_region"], f"{axis} anchor {index} region")
        _require(_half_width(region, axis) >= minimum_half_width, f"{axis} anchor {index} region is narrower than protocol")
        _require(
            _half_width(region, "x") <= maximum_half_width
            and _half_width(region, "y") <= maximum_half_width,
            f"{axis} anchor {index} region is broader than protocol",
        )
        number = _finite(item["value"], f"{axis} anchor {index} value")
        if expected_scale == "log10":
            _require(number > 0, f"{axis} log10 anchor {index} must be positive")
        display = math.log10(number) if expected_scale == "log10" else number
        _require(display not in seen_values, f"{axis} calibration repeats a tick value")
        seen_values.add(display)
        result.append({"pixel_region": region, "value": number})
    pixels = [_center(row["pixel_region"], axis) for row in result]
    _require(max(pixels) > min(pixels), f"{axis} calibration has no pixel span")
    _require(max(seen_values) > min(seen_values), f"{axis} calibration has no value span")
    return {
        "scale": expected_scale,
        "all_legible_major_ticks_included": True,
        "outermost_ticks_included": True,
        "axis_spine_region": _region(row["axis_spine_region"], f"{axis} axis spine region"),
        "anchors": result,
    }


def _fit_axis(anchors: Sequence[Mapping[str, Any]], axis: str, scale: str, pixels: np.ndarray | None = None) -> tuple[float, float, dict[str, float]]:
    source_pixels = np.array([_center(row["pixel_region"], axis) for row in anchors], dtype=np.float64) if pixels is None else pixels
    values = np.array([float(row["value"]) for row in anchors], dtype=np.float64)
    displayed = np.log10(values) if scale == "log10" else values
    centered = source_pixels - source_pixels.mean()
    denominator = float(np.sum(centered * centered))
    _require(denominator > 0, "axis calibration is singular")
    slope = float(np.sum(centered * (displayed - displayed.mean())) / denominator)
    intercept = float(displayed.mean() - slope * source_pixels.mean())
    predicted = intercept + slope * source_pixels
    residuals = predicted - displayed
    pixel_residuals = residuals / abs(slope)
    return float(intercept), float(slope), {
        "rms_pixels": float(np.sqrt(np.mean(pixel_residuals**2))),
        "maximum_pixels": float(np.max(np.abs(pixel_residuals))),
        "rms_axis_units": float(np.sqrt(np.mean(residuals**2))),
        "maximum_axis_units": float(np.max(np.abs(residuals))),
    }


def _validate_calibration(axis: Mapping[str, Any], name: str) -> dict[str, Any]:
    intercept, slope, residuals = _fit_axis(axis["anchors"], name, axis["scale"])
    _require(math.isfinite(intercept) and math.isfinite(slope) and slope != 0, f"{name} calibration fit is invalid")
    source_pixels = np.array([_center(item["pixel_region"], name) for item in axis["anchors"]])
    values = np.array([float(item["value"]) for item in axis["anchors"]])
    displayed = np.log10(values) if axis["scale"] == "log10" else values
    predicted = intercept + slope * source_pixels
    residual_pixels = (predicted - displayed) / abs(slope)
    stated_pixel_uncertainty = np.array(
        [max(_half_width(item["pixel_region"], name), 0.5) for item in axis["anchors"]]
    )
    maximum = float(np.max(np.abs(residual_pixels) / stated_pixel_uncertainty))
    _require(maximum <= 2.0 + 1e-12, f"{name} calibration standardized residual exceeds protocol")
    order = np.argsort(source_pixels)
    ordered_values = displayed[order]
    differences = np.diff(ordered_values)
    _require(
        np.all(differences > 0) or np.all(differences < 0),
        f"{name} calibration tick values are not monotonic in pixel order",
    )
    _require(np.sign(differences[0]) == np.sign(slope), f"{name} calibration orientation is inconsistent")
    transformed = list(predicted)
    expected = list(displayed)
    _require(all(math.isfinite(item) for item in transformed), f"{name} calibration inverse check fails")
    return {
        "intercept": intercept,
        "slope": slope,
        "residuals": {**residuals, "maximum_standardized_anchor_residual": maximum},
        "inverse_checks": list(zip(transformed, expected)),
    }


def _check_authoritative_x(
    marker: Mapping[str, float], authoritative_x: float, calibration: Mapping[str, Any], axis: Mapping[str, Any]
) -> None:
    """Reject a marker whose plotted x position cannot be its stated command."""
    slope = calibration["slope"]
    displayed = calibration["intercept"] + slope * np.array(
        [marker["x_min"], marker["x_max"]], dtype=np.float64
    )
    lower, upper = sorted(float(item) for item in displayed)
    # Calibration residual is measurement uncertainty, not a license to move a
    # marker across the graph.  The two-residual allowance handles raster axes.
    allowance = 2.0 * calibration["residuals"]["maximum_axis_units"]
    expected = math.log10(authoritative_x) if axis["scale"] == "log10" else authoritative_x
    _require(lower - allowance <= expected <= upper + allowance, "digitized x conflicts with authoritative command")


def _validate_biological_error(value: Any, expected_kind: str) -> dict[str, Any]:
    if value is None:
        return {"status": "not_reported"}
    _require(isinstance(value, Mapping), "biological error must be an object or null")
    status = value.get("status")
    if status in {"unavailable", "not_reported"}:
        _strict_object(value, {"status"}, "biological error")
        return {"status": status}
    if status == "available":
        row = _strict_object(value, {"kind", "lower_endpoint_region", "status", "upper_endpoint_region"}, "biological error")
        _require(row["kind"] == expected_kind, "biological error kind differs from the source protocol")
        return {
            "status": status,
            "kind": row["kind"],
            "lower_endpoint_region": _region(row["lower_endpoint_region"], "lower endpoint region"),
            "upper_endpoint_region": _region(row["upper_endpoint_region"], "upper endpoint region"),
        }
    if status == "one_sided":
        row = _strict_object(value, {"endpoint_region", "kind", "side", "status"}, "biological error")
        _require(
            row["side"] in {"lower", "upper"} and row["kind"] == expected_kind,
            "one-sided biological error differs from the source protocol",
        )
        return {"status": status, "kind": row["kind"], "side": row["side"], "endpoint_region": _region(row["endpoint_region"], "endpoint region")}
    raise PopulationCurveDigitizationError("biological error status is invalid")


def _validate_asset_binding(root: Path, value: Any, authority: Mapping[str, Any]) -> dict[str, Any]:
    row = _strict_object(value, {"asset_id", "image", "receipt"}, "asset binding")
    asset_id = row["asset_id"]
    _require(isinstance(asset_id, str) and asset_id in authority["assets"], "asset id is not in the manifest")
    source = authority["assets"][asset_id]
    acquisition = source.get("full_resolution_acquisition")
    _require(isinstance(acquisition, Mapping), "asset lacks an acquired full-resolution authority")
    receipt_row = _strict_object(
        row["receipt"], {"file_sha256", "path", "self_sha256"}, "asset receipt"
    )
    receipt_path_text, receipt_path = _repository_file(root, receipt_row["path"], "asset receipt")
    _require(receipt_path_text == acquisition.get("local_receipt_path"), "receipt path differs from manifest authority")
    _require(
        _file_digest(receipt_path) == _sha256(receipt_row["file_sha256"], "asset receipt file digest"),
        "receipt file digest mismatch",
    )
    _require(
        receipt_row["file_sha256"] == acquisition.get("receipt_file_sha256"),
        "receipt file digest differs from manifest authority",
    )
    image_row = _strict_object(row["image"], {"path", "pixel_sha256", "sha256"}, "asset image")
    image_path_text, image_path = _repository_file(root, image_row["path"], "asset image")
    _require(image_path_text == acquisition.get("local_image_path"), "image path differs from manifest authority")
    _require(_file_digest(image_path) == _sha256(image_row["sha256"], "asset image digest"), "image file digest mismatch")
    # The receipt carries a repository-relative image path.  Pass the resolved
    # path explicitly so validation does not depend on the process cwd.
    receipt = pmc_tile_asset.verify_receipt(receipt_path, image_path)
    _require(
        receipt.get("sha256") == _sha256(receipt_row["self_sha256"], "asset receipt self digest")
        == acquisition.get("receipt_self_sha256"),
        "receipt self digest differs from manifest authority",
    )
    assembled = receipt.get("assembled_image")
    _require(isinstance(assembled, Mapping), "receipt assembled image is invalid")
    _require(image_row["sha256"] == acquisition.get("assembled_image_sha256") == assembled.get("sha256"), "assembled image digest differs from authority")
    _require(image_row["pixel_sha256"] == acquisition.get("pixel_sha256") == assembled.get("pixel_sha256"), "image pixel digest differs from authority")
    _require([assembled.get("width"), assembled.get("height")] == source.get("full_resolution_pixels"), "original image dimensions differ from authority")
    _require(
        receipt["manifest"]["url"] == source.get("tile_manifest_url")
        and receipt["manifest"]["sha256"] == acquisition.get("manifest_sha256")
        and receipt["grid"]["tile_count"] == acquisition.get("tile_count")
        and receipt["view"]["sId"] == source.get("tile_view_id")
        and receipt["source"]["satellite"] == source.get("tile_satellite"),
        "receipt source metadata differs from manifest authority",
    )
    return {
        "asset_id": asset_id,
        "receipt": {
            "path": receipt_path_text,
            "file_sha256": receipt_row["file_sha256"],
            "self_sha256": receipt_row["self_sha256"],
        },
        "image": {
            "path": image_path_text,
            "sha256": image_row["sha256"],
            "pixel_sha256": image_row["pixel_sha256"],
            "width": assembled["width"],
            "height": assembled["height"],
        },
    }


def _validate_provenance(root: Path, value: Any) -> dict[str, Any]:
    row = _strict_object(
        value,
        {
            "annotation_artifact",
            "blind_to_other_extractions",
            "source_pixels_resampled",
            "tool",
            "workflow_id",
        },
        "extraction provenance",
    )
    _require(
        isinstance(row["workflow_id"], str)
        and _IDENTIFIER.fullmatch(row["workflow_id"]) is not None,
        "extraction workflow id is invalid",
    )
    _require(row["blind_to_other_extractions"] is True, "extraction is not blind")
    _require(row["source_pixels_resampled"] is False, "extraction used resampled source pixels")
    tool = _strict_object(row["tool"], {"name", "version"}, "extraction tool")
    _require(
        isinstance(tool["name"], str)
        and tool["name"]
        and isinstance(tool["version"], str)
        and tool["version"],
        "extraction tool identity is invalid",
    )
    annotation, _ = _binding(root, row["annotation_artifact"], "annotation artifact")
    return {
        "workflow_id": row["workflow_id"],
        "blind_to_other_extractions": True,
        "source_pixels_resampled": False,
        "tool": dict(tool),
        "annotation_artifact": annotation,
    }


def validate_extraction_record(record: Mapping[str, Any], authority: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Strictly validate one immutable raw extraction record.

    The returned object is normalized but retains no information beyond what
    was explicitly supplied by the extractor.
    """
    root = Path(root).resolve()
    row = _strict_object(record, {"asset", "extractor_id", "measurement", "panel", "protocol", "provenance", "record_id", "schema", "status", "unavailable_reason"}, "extraction record")
    _require(row["schema"] == RECORD_SCHEMA, "extraction record schema is invalid")
    _require(isinstance(row["record_id"], str) and _IDENTIFIER.fullmatch(row["record_id"]) is not None, "record id is invalid")
    _require(isinstance(row["extractor_id"], str) and _IDENTIFIER.fullmatch(row["extractor_id"]) is not None, "extractor id must be a non-personal opaque identifier")
    protocol_binding, _ = _binding(root, row["protocol"], "record protocol")
    _require(protocol_binding == authority["binding"], "record protocol binding differs from loaded protocol")
    provenance = _validate_provenance(root, row["provenance"])
    asset = _validate_asset_binding(root, row["asset"], authority)
    panel = _strict_object(row["panel"], {"bounds", "id", "series_id"}, "panel")
    panel_key = (asset["asset_id"], panel["id"])
    _require(panel_key in authority["panels"], "panel is not eligible")
    expected = authority["panels"][panel_key]
    _require(panel["series_id"] == expected.get("series_identity"), "series identity differs from protocol")
    bounds = _strict_object(panel["bounds"], {"x_max", "x_min", "y_max", "y_min"}, "panel bounds")
    normalized_bounds = {key: _finite(value, f"panel bounds {key}") for key, value in bounds.items()}
    _require(0 <= normalized_bounds["x_min"] < normalized_bounds["x_max"] <= asset["image"]["width"], "panel x bounds are outside original image")
    _require(0 <= normalized_bounds["y_min"] < normalized_bounds["y_max"] <= asset["image"]["height"], "panel y bounds are outside original image")
    expected_bounds = expected.get("panel_bounds_original_pixels")
    _require(
        isinstance(expected_bounds, list)
        and len(expected_bounds) == 4
        and normalized_bounds
        == {
            "x_min": float(expected_bounds[0]),
            "y_min": float(expected_bounds[1]),
            "x_max": float(expected_bounds[2]),
            "y_max": float(expected_bounds[3]),
        },
        "panel bounds differ from the preregistered original-pixel bounds",
    )
    _require(row["status"] in {"available", "unavailable"}, "record status is invalid")
    if row["status"] == "unavailable":
        _require(row["measurement"] is None and row["unavailable_reason"] in _UNAVAILABLE_REASONS, "unavailable record must contain only a preregistered unavailable reason")
        return {"schema": RECORD_SCHEMA, "record_id": row["record_id"], "extractor_id": row["extractor_id"], "protocol": protocol_binding, "provenance": provenance, "asset": asset, "panel": {"id": panel["id"], "series_id": panel["series_id"], "bounds": normalized_bounds}, "status": "unavailable", "unavailable_reason": row["unavailable_reason"]}
    _require(row["unavailable_reason"] is None and isinstance(row["measurement"], Mapping), "available record fields are invalid")
    measurement = _strict_object(row["measurement"], {"points", "x_axis", "y_axis"}, "measurement")
    minimum_half = authority["document"]["axis_calibration"]["tick_pixel_region_minimum_half_width"]
    maximum_half = authority["document"]["axis_calibration"]["tick_pixel_region_maximum_half_width"]
    x_axis = _validate_axis(measurement["x_axis"], "x", expected["x_scale"], minimum_half, maximum_half)
    y_axis = _validate_axis(measurement["y_axis"], "y", expected["y_scale"], minimum_half, maximum_half)
    for name, axis in (("x", x_axis), ("y", y_axis)):
        _inside_panel(axis["axis_spine_region"], normalized_bounds, f"{name} axis spine")
        for index, anchor in enumerate(axis["anchors"]):
            _inside_panel(anchor["pixel_region"], normalized_bounds, f"{name} anchor {index}")
        anchor_centers = [_center(item["pixel_region"], name) for item in axis["anchors"]]
        spine = axis["axis_spine_region"]
        spine_span = spine[f"{name}_max"] - spine[f"{name}_min"]
        _require(
            spine_span > 0
            and (max(anchor_centers) - min(anchor_centers)) / spine_span
            >= authority["document"]["axis_calibration"]["minimum_calibrated_axis_span_fraction"],
            f"{name} calibration does not cover the preregistered axis span",
        )
    x_calibration = _validate_calibration(x_axis, "x")
    y_calibration = _validate_calibration(y_axis, "y")
    points = measurement["points"]
    _require(isinstance(points, list) and points, "measurement has no points")
    normalized_points: list[dict[str, Any]] = []
    command_ids: set[str] = set()
    observed_authoritative_x: set[float] = set()
    x_authority_mode = expected.get("x_authority_mode")
    _require(x_authority_mode in {"published_command", "digitized_source_marker"}, "panel x authority mode is invalid")
    for index, point in enumerate(points):
        item = _strict_object(
            point,
            {
                "authoritative_x",
                "biological_error",
                "command_id",
                "marker_center_region",
                "occlusion",
                "status",
                "unavailable_reason",
            },
            f"point {index}",
        )
        command_id = item["command_id"]
        _require(isinstance(command_id, str) and _IDENTIFIER.fullmatch(command_id) is not None and command_id not in command_ids, f"point {index} command id is invalid")
        command_ids.add(command_id)
        if x_authority_mode == "published_command":
            authoritative_x: float | None = _finite(item["authoritative_x"], f"point {index} authoritative x")
            if expected["x_scale"] == "log10":
                _require(authoritative_x > 0, f"point {index} log10 authoritative x must be positive")
            declared = expected.get("authoritative_x_values")
            _require(
                isinstance(declared, list) and authoritative_x in declared,
                f"point {index} authoritative x is not a published command",
            )
            _require(authoritative_x not in observed_authoritative_x, f"point {index} repeats an authoritative command")
            observed_authoritative_x.add(authoritative_x)
        else:
            _require(item["authoritative_x"] is None, f"point {index} invents authority for a digitized source-marker x")
            authoritative_x = None
        _require(item["status"] in {"available", "unavailable"}, f"point {index} status is invalid")
        if item["status"] == "unavailable":
            _require(
                x_authority_mode == "published_command"
                and item["marker_center_region"] is None
                and item["occlusion"] is None
                and item["biological_error"] is None
                and item["unavailable_reason"] in _UNAVAILABLE_REASONS,
                f"point {index} unavailable fields are invalid",
            )
            normalized_points.append(
                {
                    "command_id": command_id,
                    "authoritative_x": authoritative_x,
                    "status": "unavailable",
                    "unavailable_reason": item["unavailable_reason"],
                    "marker_center_region": None,
                    "biological_error": None,
                }
            )
            continue
        _require(item["unavailable_reason"] is None, f"point {index} available point has an unavailable reason")
        occlusion = _strict_object(
            item["occlusion"],
            {
                "bounded_without_curve_interpolation",
                "opposing_marker_edges_visible",
                "partial",
                "unique_series_identity",
            },
            f"point {index} occlusion",
        )
        _require(
            occlusion["bounded_without_curve_interpolation"] is True
            and occlusion["unique_series_identity"] is True
            and type(occlusion["partial"]) is bool
            and type(occlusion["opposing_marker_edges_visible"]) is bool
            and (not occlusion["partial"] or occlusion["opposing_marker_edges_visible"]),
            f"point {index} occlusion criteria fail",
        )
        marker = _region(item["marker_center_region"], f"point {index} marker center")
        _require(_half_width(marker, "x") >= authority["document"]["point_measurement"]["marker_center_minimum_half_width_pixels"], f"point {index} marker x region is narrower than protocol")
        _require(_half_width(marker, "y") >= authority["document"]["point_measurement"]["marker_center_minimum_half_width_pixels"], f"point {index} marker y region is narrower than protocol")
        _require(
            _half_width(marker, "x")
            <= authority["document"]["point_measurement"]["marker_center_maximum_half_width_pixels"]
            and _half_width(marker, "y")
            <= authority["document"]["point_measurement"]["marker_center_maximum_half_width_pixels"],
            f"point {index} marker region is broader than protocol",
        )
        _inside_panel(marker, normalized_bounds, f"point {index} marker")
        if authoritative_x is not None:
            _check_authoritative_x(marker, authoritative_x, x_calibration, x_axis)
        biological_error = _validate_biological_error(
            item["biological_error"], expected["biological_error_bar"]
        )
        for endpoint_key in ("lower_endpoint_region", "upper_endpoint_region", "endpoint_region"):
            endpoint = biological_error.get(endpoint_key)
            if endpoint is not None:
                _require(_half_width(endpoint, "y") >= authority["document"]["point_measurement"]["marker_center_minimum_half_width_pixels"], f"point {index} biological error endpoint is narrower than protocol")
                _inside_panel(endpoint, normalized_bounds, f"point {index} biological error endpoint")
                _require(
                    _half_width(endpoint, "x")
                    <= authority["document"]["error_bars"]["endpoint_center_maximum_half_width_pixels"]
                    and _half_width(endpoint, "y")
                    <= authority["document"]["error_bars"]["endpoint_center_maximum_half_width_pixels"],
                    f"point {index} biological error endpoint is broader than protocol",
                )
                _require(
                    endpoint["x_min"] <= marker["x_max"]
                    and marker["x_min"] <= endpoint["x_max"],
                    f"point {index} biological error endpoint does not overlap marker x",
                )
        if biological_error["status"] == "available":
            marker_value = y_calibration["intercept"] + y_calibration["slope"] * _center(marker, "y")
            lower_value = y_calibration["intercept"] + y_calibration["slope"] * _center(
                biological_error["lower_endpoint_region"], "y"
            )
            upper_value = y_calibration["intercept"] + y_calibration["slope"] * _center(
                biological_error["upper_endpoint_region"], "y"
            )
            _require(lower_value <= marker_value <= upper_value, f"point {index} biological error endpoints are misordered")
        elif biological_error["status"] == "one_sided":
            marker_value = y_calibration["intercept"] + y_calibration["slope"] * _center(marker, "y")
            endpoint_value = y_calibration["intercept"] + y_calibration["slope"] * _center(
                biological_error["endpoint_region"], "y"
            )
            if biological_error["side"] == "lower":
                _require(endpoint_value <= marker_value, f"point {index} lower error endpoint is misordered")
            else:
                _require(endpoint_value >= marker_value, f"point {index} upper error endpoint is misordered")
        normalized_points.append(
            {
                "command_id": command_id,
                "authoritative_x": authoritative_x,
                "status": "available",
                "unavailable_reason": None,
                "marker_center_region": marker,
                "occlusion": dict(occlusion),
                "biological_error": biological_error,
            }
        )
    if x_authority_mode == "published_command":
        _require(
            observed_authoritative_x == {float(item) for item in expected["authoritative_x_values"]},
            "published command set is incomplete",
        )
    calibrated_order = sorted(
        normalized_points,
        key=lambda point: (
            point["authoritative_x"]
            if point["authoritative_x"] is not None
            else x_calibration["intercept"]
            + x_calibration["slope"] * _center(point["marker_center_region"], "x")
        ),
    )
    _require(
        normalized_points == calibrated_order
        and [point["command_id"] for point in normalized_points]
        == [f"command_{index:03d}" for index in range(1, len(normalized_points) + 1)],
        "points do not follow the preregistered calibrated-x command id order",
    )
    return {"schema": RECORD_SCHEMA, "record_id": row["record_id"], "extractor_id": row["extractor_id"], "protocol": protocol_binding, "provenance": provenance, "asset": asset, "panel": {"id": panel["id"], "series_id": panel["series_id"], "bounds": normalized_bounds}, "status": "available", "unavailable_reason": None, "measurement": {"x_axis": x_axis, "y_axis": y_axis, "points": normalized_points}}


def _seed(asset_sha256: str, panel_id: str, series_id: str, command_id: str) -> int:
    material = "|".join((asset_sha256, panel_id, series_id, command_id)).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _sample_axis(axis: Mapping[str, Any], coordinate: str, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = axis["anchors"]
    lows = np.array([item["pixel_region"][f"{coordinate}_min"] for item in anchors], dtype=np.float64)
    highs = np.array([item["pixel_region"][f"{coordinate}_max"] for item in anchors], dtype=np.float64)
    pixels = rng.uniform(lows, highs, size=(draws, len(anchors)))
    values = np.array([item["value"] for item in anchors], dtype=np.float64)
    display = np.log10(values) if axis["scale"] == "log10" else values
    x_mean = pixels.mean(axis=1)
    centered = pixels - x_mean[:, None]
    denominator = np.sum(centered * centered, axis=1)
    _require(np.all(denominator > 0), "Monte Carlo calibration is singular")
    slope = np.sum(centered * (display - display.mean()), axis=1) / denominator
    intercept = display.mean() - slope * x_mean
    nominal_slope = _fit_axis(anchors, coordinate, axis["scale"])[1]
    _require(
        np.all(np.isfinite(slope))
        and np.all(np.isfinite(intercept))
        and np.all(np.sign(slope) == np.sign(nominal_slope)),
        "Monte Carlo calibration reverses or invalidates the axis",
    )
    return intercept, slope, pixels


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.quantile(values, 0.5, method="linear")),
        "standard_uncertainty": float(np.std(values, ddof=1)),
        "q025": float(np.quantile(values, 0.025, method="linear")),
        "q975": float(np.quantile(values, 0.975, method="linear")),
    }


def _display_to_value(value: np.ndarray, scale: str) -> np.ndarray:
    return np.power(10.0, value) if scale == "log10" else value


def _one_point_samples(
    record: Mapping[str, Any], point: Mapping[str, Any], authority: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    draws = authority["document"]["digitization_uncertainty"]["monte_carlo_draws"]
    _require(type(draws) is int and 1 <= draws <= 1_000_000, "protocol Monte Carlo draw count is invalid")
    seed = _seed(record["asset"]["image"]["sha256"], record["panel"]["id"], record["panel"]["series_id"], point["command_id"])
    x_axis, y_axis = record["measurement"]["x_axis"], record["measurement"]["y_axis"]
    rng = np.random.Generator(np.random.PCG64(seed))
    x_intercept, x_slope, _ = _sample_axis(x_axis, "x", draws, rng)
    y_intercept, y_slope, _ = _sample_axis(y_axis, "y", draws, rng)
    x_nominal_i, x_nominal_s, x_nominal_residuals = _fit_axis(
        x_axis["anchors"], "x", x_axis["scale"]
    )
    y_nominal_i, y_nominal_s, y_nominal_residuals = _fit_axis(
        y_axis["anchors"], "y", y_axis["scale"]
    )
    x_residual_noise = rng.normal(0.0, x_nominal_residuals["rms_axis_units"], draws)
    y_residual_noise = rng.normal(0.0, y_nominal_residuals["rms_axis_units"], draws)
    marker = point["marker_center_region"]
    marker_x = rng.uniform(marker["x_min"], marker["x_max"], draws)
    marker_y = rng.uniform(marker["y_min"], marker["y_max"], draws)
    total_x = _display_to_value(
        x_intercept + x_slope * marker_x + x_residual_noise, x_axis["scale"]
    )
    total_y = _display_to_value(
        y_intercept + y_slope * marker_y + y_residual_noise, y_axis["scale"]
    )
    axis_only_y = _display_to_value(
        y_intercept + y_slope * _center(marker, "y") + y_residual_noise,
        y_axis["scale"],
    )
    marker_only_y = _display_to_value(y_nominal_i + y_nominal_s * marker_y, y_axis["scale"])
    error = point["biological_error"]
    biological: dict[str, Any] = dict(error)
    samples = {"x": total_x, "y": total_y}
    if error["status"] == "available":
        lower = error["lower_endpoint_region"]
        upper = error["upper_endpoint_region"]
        lower_y = rng.uniform(lower["y_min"], lower["y_max"], draws)
        upper_y = rng.uniform(upper["y_min"], upper["y_max"], draws)
        samples["lower_endpoint"] = _display_to_value(
            y_intercept + y_slope * lower_y + y_residual_noise, y_axis["scale"]
        )
        samples["upper_endpoint"] = _display_to_value(
            y_intercept + y_slope * upper_y + y_residual_noise, y_axis["scale"]
        )
        biological["lower_endpoint_digitization"] = _quantiles(samples["lower_endpoint"])
        biological["upper_endpoint_digitization"] = _quantiles(samples["upper_endpoint"])
    elif error["status"] == "one_sided":
        endpoint = error["endpoint_region"]
        endpoint_y = rng.uniform(endpoint["y_min"], endpoint["y_max"], draws)
        samples["endpoint"] = _display_to_value(
            y_intercept + y_slope * endpoint_y + y_residual_noise, y_axis["scale"]
        )
        biological["endpoint_digitization"] = _quantiles(samples["endpoint"])
    return {
        "command_id": point["command_id"],
        "authoritative_x": point["authoritative_x"],
        "status": "available",
        "unavailable_reason": None,
        "digitized_x": _quantiles(total_x),
        "digitized_y": _quantiles(total_y),
        "digitization_uncertainty": {
            "axis_calibration_component": float(np.std(axis_only_y, ddof=1)),
            "marker_or_endpoint_component": float(np.std(marker_only_y, ddof=1)),
            "between_extractor_component": None,
        },
        "biological_error": biological,
        "seed": f"{seed:032x}",
    }, samples


def _one_point(record: Mapping[str, Any], point: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    output, _ = _one_point_samples(record, point, authority)
    return output


def _agreement_statistic(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    denominator = math.hypot(first["standard_uncertainty"], second["standard_uncertainty"])
    if denominator == 0:
        return math.inf if first["median"] != second["median"] else 0.0
    return abs(first["median"] - second["median"]) / denominator


def digitize_record(record: Mapping[str, Any], authority: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Validate a record and deterministically propagate its stated pixel bounds."""
    normalized = validate_extraction_record(record, authority, root=root)
    core: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "scientific_verdict": None,
        "optimization_command": None,
        "optimization_allowed": False,
        "promotion_status": "measurement_only",
        "raw_record_sha256": digest(record),
        "runtime": _runtime(),
        "record": {key: normalized[key] for key in ("asset", "extractor_id", "panel", "protocol", "provenance", "record_id", "status", "unavailable_reason")},
    }
    if normalized["status"] == "unavailable":
        core["points"] = []
    else:
        core["points"] = [
            (
                {
                    "command_id": point["command_id"],
                    "authoritative_x": point["authoritative_x"],
                    "status": "unavailable",
                    "unavailable_reason": point["unavailable_reason"],
                    "digitized_x": None,
                    "digitized_y": None,
                    "digitization_uncertainty": None,
                    "biological_error": None,
                    "seed": None,
                }
                if point["status"] == "unavailable"
                else _one_point(normalized, point, authority)
            )
            for point in normalized["measurement"]["points"]
        ]
    core["sha256"] = digest(core)
    return core


def compare_blind_extractions(first: Mapping[str, Any], second: Mapping[str, Any], authority: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Compare two independently digitized records without adjudicating them."""
    left = digitize_record(first, authority, root=root)
    right = digitize_record(second, authority, root=root)
    _require(left["record"]["extractor_id"] != right["record"]["extractor_id"], "blind extractions require distinct extractor ids")
    _require(
        left["record"]["provenance"]["workflow_id"]
        != right["record"]["provenance"]["workflow_id"]
        and left["record"]["provenance"]["annotation_artifact"]["sha256"]
        != right["record"]["provenance"]["annotation_artifact"]["sha256"],
        "blind extractions require distinct workflows and annotation artifacts",
    )
    for key in ("asset", "panel", "protocol"):
        _require(left["record"][key] == right["record"][key], f"blind extraction {key} differs")
    left_status = left["record"]["status"]
    right_status = right["record"]["status"]
    if left_status != "available" or right_status != "available":
        accepted = (
            left_status == right_status == "unavailable"
            and left["record"]["unavailable_reason"]
            == right["record"]["unavailable_reason"]
        )
        core = {
            "schema": COMPARISON_SCHEMA,
            "scientific_verdict": None,
            "optimization_command": None,
            "optimization_allowed": False,
            "promotion_status": "measurement_only",
            "runtime": _runtime(),
            "first": {
                "record_id": left["record"]["record_id"],
                "raw_record_sha256": left["raw_record_sha256"],
                "sha256": left["sha256"],
            },
            "second": {
                "record_id": right["record"]["record_id"],
                "raw_record_sha256": right["raw_record_sha256"],
                "sha256": right["sha256"],
            },
            "panel_availability": {
                "accepted": accepted,
                "both_available": False,
                "first": {
                    "status": left_status,
                    "unavailable_reason": left["record"]["unavailable_reason"],
                },
                "second": {
                    "status": right_status,
                    "unavailable_reason": right["record"]["unavailable_reason"],
                },
                "resolved_unavailable_reason": (
                    left["record"]["unavailable_reason"] if accepted else None
                ),
            },
            "points": [],
            "status": (
                "two_extractions_agree_panel_unavailable"
                if accepted
                else "third_blind_independent_extraction_required"
            ),
            "third_extraction_required": not accepted,
        }
        core["sha256"] = digest(core)
        return core
    left_points = {point["command_id"]: point for point in left["points"]}
    right_points = {point["command_id"]: point for point in right["points"]}
    if set(left_points) != set(right_points):
        core = {
            "schema": COMPARISON_SCHEMA,
            "scientific_verdict": None,
            "optimization_command": None,
            "optimization_allowed": False,
            "promotion_status": "measurement_only",
            "runtime": _runtime(),
            "first": {
                "record_id": left["record"]["record_id"],
                "raw_record_sha256": left["raw_record_sha256"],
                "sha256": left["sha256"],
            },
            "second": {
                "record_id": right["record"]["record_id"],
                "raw_record_sha256": right["raw_record_sha256"],
                "sha256": right["sha256"],
            },
            "panel_availability": {
                "accepted": True,
                "both_available": True,
                "first": {"status": "available", "unavailable_reason": None},
                "second": {"status": "available", "unavailable_reason": None},
                "resolved_unavailable_reason": None,
            },
            "command_set_agreement": {
                "accepted": False,
                "first": sorted(left_points),
                "second": sorted(right_points),
            },
            "points": [],
            "status": "third_blind_independent_extraction_required",
            "third_extraction_required": True,
        }
        core["sha256"] = digest(core)
        return core
    threshold = authority["document"]["agreement_and_adjudication"]["threshold"]
    raw_left = validate_extraction_record(first, authority, root=root)
    raw_right = validate_extraction_record(second, authority, root=root)
    raw_left_points = {point["command_id"]: point for point in raw_left["measurement"]["points"]}
    raw_right_points = {point["command_id"]: point for point in raw_right["measurement"]["points"]}
    rows: list[dict[str, Any]] = []
    third_required = False
    panel_protocol = authority["panels"][(
        left["record"]["asset"]["asset_id"], left["record"]["panel"]["id"]
    )]
    for command_id in sorted(left_points):
        left_point = left_points[command_id]
        right_point = right_points[command_id]
        if left_point["status"] == "unavailable" or right_point["status"] == "unavailable":
            accepted = (
                left_point["status"] == right_point["status"] == "unavailable"
                and left_point["unavailable_reason"] == right_point["unavailable_reason"]
            )
            third_required = third_required or not accepted
            rows.append(
                {
                    "command_id": command_id,
                    "normalized_disagreement": {},
                    "compared_channels": [],
                    "threshold": threshold,
                    "accepted": accepted,
                    "combined_digitized_x": None,
                    "combined_digitized_y": None,
                    "combined_digitization_uncertainty": None,
                    "biological_error": None,
                    "availability": {
                        "first": left_point["status"],
                        "second": right_point["status"],
                        "unavailable_reason": (
                            left_point["unavailable_reason"] if accepted else None
                        ),
                    },
                }
            )
            continue
        left_output, left_samples = _one_point_samples(
            raw_left, raw_left_points[command_id], authority
        )
        right_output, right_samples = _one_point_samples(
            raw_right, raw_right_points[command_id], authority
        )
        _require(left_output == left_points[command_id] and right_output == right_points[command_id], "non-deterministic digitization output")
        statistics = {
            "y": _agreement_statistic(left_output["digitized_y"], right_output["digitized_y"])
        }
        compared_channels = ["y"]
        if panel_protocol["x_authority_mode"] == "digitized_source_marker":
            statistics["x"] = _agreement_statistic(
                left_output["digitized_x"], right_output["digitized_x"]
            )
            compared_channels.append("x")
        left_bio = left_output["biological_error"]
        right_bio = right_output["biological_error"]
        endpoint_channels = ("lower_endpoint", "upper_endpoint", "endpoint")
        left_endpoint_channels = {
            channel for channel in endpoint_channels if channel in left_samples
        }
        right_endpoint_channels = {
            channel for channel in endpoint_channels if channel in right_samples
        }
        left_bio_signature = {
            key: left_bio.get(key) for key in ("status", "kind", "side")
            if key in left_bio
        }
        right_bio_signature = {
            key: right_bio.get(key) for key in ("status", "kind", "side")
            if key in right_bio
        }
        biological_compatible = (
            left_bio_signature == right_bio_signature
            and left_endpoint_channels == right_endpoint_channels
        )
        biological: dict[str, Any] = (
            dict(left_bio_signature)
            if biological_compatible
            else {
                "status": "extractor_disagreement",
                "first": left_bio_signature,
                "second": right_bio_signature,
            }
        )
        for channel, output_key in (
            ("lower_endpoint", "lower_endpoint_digitization"),
            ("upper_endpoint", "upper_endpoint_digitization"),
            ("endpoint", "endpoint_digitization"),
        ):
            if biological_compatible and channel in left_endpoint_channels:
                statistics[channel] = _agreement_statistic(
                    left_bio[output_key], right_bio[output_key]
                )
                compared_channels.append(channel)
                biological[output_key] = _quantiles(
                    np.concatenate((left_samples[channel], right_samples[channel]))
                )
        accepted = biological_compatible and all(
            statistics[channel] <= threshold for channel in compared_channels
        )
        third_required = third_required or not accepted
        y_mixture = np.concatenate((left_samples["y"], right_samples["y"]))
        x_mixture = np.concatenate((left_samples["x"], right_samples["x"]))
        rows.append({
            "command_id": command_id,
            "normalized_disagreement": statistics,
            "compared_channels": compared_channels,
            "threshold": threshold,
            "accepted": accepted,
            "combined_digitized_x": _quantiles(x_mixture),
            "combined_digitized_y": _quantiles(y_mixture),
            "combined_digitization_uncertainty": {
                "axis_calibration_component": None,
                "marker_or_endpoint_component": None,
                "between_extractor_component": abs(
                    left_output["digitized_y"]["median"] - right_output["digitized_y"]["median"]
                ) / 2.0,
            },
            "biological_error": biological,
            "biological_error_agreement": {
                "accepted": biological_compatible,
                "first": left_bio_signature,
                "second": right_bio_signature,
            },
            "availability": {"first": "available", "second": "available", "unavailable_reason": None},
        })
    core = {
        "schema": COMPARISON_SCHEMA,
        "scientific_verdict": None,
        "optimization_command": None,
        "optimization_allowed": False,
        "promotion_status": "measurement_only",
        "runtime": _runtime(),
        "first": {
            "record_id": left["record"]["record_id"],
            "raw_record_sha256": left["raw_record_sha256"],
            "sha256": left["sha256"],
        },
        "second": {
            "record_id": right["record"]["record_id"],
            "raw_record_sha256": right["raw_record_sha256"],
            "sha256": right["sha256"],
        },
        "panel_availability": {
            "accepted": True,
            "both_available": True,
            "first": {"status": "available", "unavailable_reason": None},
            "second": {"status": "available", "unavailable_reason": None},
            "resolved_unavailable_reason": None,
        },
        "command_set_agreement": {
            "accepted": True,
            "first": sorted(left_points),
            "second": sorted(right_points),
        },
        "points": rows,
        "status": (
            "third_blind_independent_extraction_required"
            if third_required
            else "two_extractions_agree"
        ),
        "third_extraction_required": third_required,
    }
    core["sha256"] = digest(core)
    return core


def adjudicate_three_extractions(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    third: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Resolve a failed blind pair only when the third agrees with exactly one original."""
    ab = compare_blind_extractions(first, second, authority, root=root)
    _require(ab["third_extraction_required"] is True, "third extraction is not authorized when the blind pair agrees")
    ac = compare_blind_extractions(first, third, authority, root=root)
    bc = compare_blind_extractions(second, third, authority, root=root)
    comparisons = {
        "first_second": ab,
        "first_third": ac,
        "second_third": bc,
    }
    if any(not item["panel_availability"]["both_available"] for item in comparisons.values()):
        accepted_pairs = [
            name
            for name in ("first_third", "second_third")
            if comparisons[name]["third_extraction_required"] is False
        ]
        selected = accepted_pairs[0] if len(accepted_pairs) == 1 else None
        selected_comparison = comparisons[selected] if selected is not None else None
        rows = []
        if selected_comparison is not None and selected_comparison["panel_availability"]["both_available"]:
            for point in selected_comparison["points"]:
                rows.append(
                    {
                        "command_id": point["command_id"],
                        "resolved_pair": selected,
                        "status": "resolved",
                        "measurement": point,
                        "pair_acceptance": {
                            name: not comparisons[name]["third_extraction_required"]
                            for name in comparisons
                        },
                    }
                )
        bindings = []
        for record in (first, second, third):
            output = digitize_record(record, authority, root=root)
            bindings.append(
                {
                    "record_id": output["record"]["record_id"],
                    "raw_record_sha256": output["raw_record_sha256"],
                    "sha256": output["sha256"],
                }
            )
        unresolved = selected is None
        core = {
            "schema": ADJUDICATION_SCHEMA,
            "scientific_verdict": None,
            "optimization_command": None,
            "optimization_allowed": False,
            "promotion_status": "measurement_only",
            "runtime": _runtime(),
            "records": bindings,
            "pair_comparisons": {
                name: item["sha256"] for name, item in comparisons.items()
            },
            "panel_resolution": {
                "status": "unresolved" if unresolved else "resolved",
                "resolved_pair": selected,
                "selected_panel_availability": (
                    selected_comparison["panel_availability"]
                    if selected_comparison is not None
                    else None
                ),
                "pair_acceptance": {
                    name: not item["third_extraction_required"]
                    for name, item in comparisons.items()
                },
            },
            "points": rows,
            "status": "three_extractions_unresolved" if unresolved else "three_extractions_resolved",
            "unresolved": unresolved,
        }
        core["sha256"] = digest(core)
        return core
    pair_rows = {
        "first_second": {row["command_id"]: row for row in ab["points"]},
        "first_third": {row["command_id"]: row for row in ac["points"]},
        "second_third": {row["command_id"]: row for row in bc["points"]},
    }
    rows: list[dict[str, Any]] = []
    unresolved = False
    for command_id in sorted(pair_rows["first_second"]):
        original = pair_rows["first_second"][command_id]
        if original["accepted"]:
            selected = "first_second"
        else:
            candidates = [
                name
                for name in ("first_third", "second_third")
                if pair_rows[name][command_id]["accepted"]
            ]
            selected = candidates[0] if len(candidates) == 1 else None
        unresolved = unresolved or selected is None
        rows.append(
            {
                "command_id": command_id,
                "resolved_pair": selected,
                "status": "resolved" if selected is not None else "unresolved",
                "measurement": pair_rows[selected][command_id] if selected is not None else None,
                "pair_acceptance": {
                    name: pair_rows[name][command_id]["accepted"] for name in pair_rows
                },
            }
        )
    bindings = []
    for record in (first, second, third):
        output = digitize_record(record, authority, root=root)
        bindings.append(
            {
                "record_id": output["record"]["record_id"],
                "raw_record_sha256": output["raw_record_sha256"],
                "sha256": output["sha256"],
            }
        )
    core = {
        "schema": ADJUDICATION_SCHEMA,
        "scientific_verdict": None,
        "optimization_command": None,
        "optimization_allowed": False,
        "promotion_status": "measurement_only",
        "runtime": _runtime(),
        "records": bindings,
        "pair_comparisons": {
            "first_second": ab["sha256"],
            "first_third": ac["sha256"],
            "second_third": bc["sha256"],
        },
        "panel_resolution": {
            "status": "resolved" if not unresolved else "unresolved",
            "resolved_pair": "per_command",
            "selected_panel_availability": None,
            "pair_acceptance": None,
        },
        "points": rows,
        "status": "three_extractions_unresolved" if unresolved else "three_extractions_resolved",
        "unresolved": unresolved,
    }
    core["sha256"] = digest(core)
    return core


def _complete_cliques(
    vertices: Sequence[str], accepted_edges: Mapping[tuple[str, str], bool]
) -> list[tuple[str, ...]]:
    """Return every largest complete agreement clique of at least three vertices."""

    candidates: list[tuple[str, ...]] = []
    for size in range(3, len(vertices) + 1):
        for members in combinations(vertices, size):
            if all(accepted_edges[tuple(sorted(pair))] for pair in combinations(members, 2)):
                candidates.append(members)
    if not candidates:
        return []
    maximum_size = max(len(members) for members in candidates)
    return [members for members in candidates if len(members) == maximum_size]


def adjudicate_four_extractions(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    third: Mapping[str, Any],
    fourth: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Apply the sealed four-extractor consensus protocol without inference.

    This intentionally does not reuse the three-extractor adjudicator.  Four
    records have a different preregistered rule: panel and command-set votes
    need three supporters, while each individual point needs one unique
    maximum complete-agreement clique of at least three records.
    """

    supplied = (first, second, third, fourth)
    outputs = [digitize_record(record, authority, root=root) for record in supplied]
    ordered = sorted(
        (
            output["record"]["extractor_id"],
            record,
            output,
        )
        for record, output in zip(supplied, outputs)
    )
    extractor_ids = [item[0] for item in ordered]
    _require(len(set(extractor_ids)) == 4, "four blind extractions require distinct extractor ids")
    for _, _, output in ordered[1:]:
        for key in ("asset", "panel", "protocol"):
            _require(
                output["record"][key] == ordered[0][2]["record"][key],
                f"blind extraction {key} differs",
            )

    pair_results: dict[tuple[str, str], dict[str, Any]] = {}
    for left, right in combinations(ordered, 2):
        pair = (left[0], right[0])
        pair_results[pair] = compare_blind_extractions(
            left[1], right[1], authority, root=root
        )

    bindings = [
        {
            "extractor_id": extractor_id,
            "record_id": output["record"]["record_id"],
            "raw_record_sha256": output["raw_record_sha256"],
            "sha256": output["sha256"],
        }
        for extractor_id, _, output in ordered
    ]
    comparison_bindings = [
        {
            "extractor_ids": list(pair),
            "sha256": pair_results[pair]["sha256"],
        }
        for pair in sorted(pair_results)
    ]
    available = [
        (extractor_id, output)
        for extractor_id, _, output in ordered
        if output["record"]["status"] == "available"
    ]
    unavailable_votes = [
        {
            "extractor_id": extractor_id,
            "unavailable_reason": output["record"]["unavailable_reason"],
        }
        for extractor_id, _, output in ordered
        if output["record"]["status"] == "unavailable"
    ]
    panel_votes = {
        "available_extractor_ids": [extractor_id for extractor_id, _ in available],
        "unavailable_votes": unavailable_votes,
    }
    base = {
        "schema": FOUR_WAY_ADJUDICATION_SCHEMA,
        "scientific_verdict": None,
        "optimization_command": None,
        "optimization_allowed": False,
        "promotion_status": "measurement_only",
        "runtime": _runtime(),
        "records": bindings,
        "pair_comparisons": comparison_bindings,
        "panel_status_vote": panel_votes,
    }

    if len(unavailable_votes) >= 3:
        unavailable_reasons = sorted(
            {vote["unavailable_reason"] for vote in unavailable_votes}
        )
        mixed = len(unavailable_reasons) > 1
        core = {
            **base,
            "panel_resolution": {
                "status": (
                    "majority_unavailable_mixed_reasons"
                    if mixed
                    else "resolved_unavailable"
                ),
                "selected_status": "unavailable",
                "unavailable_reasons": unavailable_reasons,
            },
            "command_set_resolution": {
                "status": "not_applicable_panel_unavailable",
                "command_ids": [],
                "votes": [],
            },
            "points": [],
            "status": "four_extractions_resolved",
            "unresolved": False,
        }
        core["sha256"] = digest(core)
        return core

    if len(available) < 3:
        core = {
            **base,
            "panel_resolution": {
                "status": "unresolved",
                "selected_status": None,
                "unavailable_reasons": sorted(
                    {vote["unavailable_reason"] for vote in unavailable_votes}
                ),
            },
            "command_set_resolution": {
                "status": "not_applicable_panel_status_unresolved",
                "command_ids": [],
                "votes": [],
            },
            "points": [],
            "status": "four_extractions_unresolved",
            "unresolved": True,
        }
        core["sha256"] = digest(core)
        return core

    command_votes: dict[tuple[str, ...], list[str]] = {}
    for extractor_id, output in available:
        command_ids = tuple(sorted(point["command_id"] for point in output["points"]))
        command_votes.setdefault(command_ids, []).append(extractor_id)
    vote_rows = [
        {
            "command_ids": list(command_ids),
            "extractor_ids": sorted(voters),
        }
        for command_ids, voters in sorted(command_votes.items())
    ]
    supported_sets = [
        command_ids
        for command_ids, voters in command_votes.items()
        if len(voters) >= 3
    ]
    if len(supported_sets) != 1:
        core = {
            **base,
            "panel_resolution": {
                "status": "resolved_available",
                "selected_status": "available",
                "unavailable_reasons": [],
            },
            "command_set_resolution": {
                "status": "unresolved",
                "command_ids": [],
                "votes": vote_rows,
            },
            "points": [],
            "status": "four_extractions_unresolved",
            "unresolved": True,
        }
        core["sha256"] = digest(core)
        return core

    selected_commands = supported_sets[0]
    supporting_extractors = sorted(command_votes[selected_commands])
    pair_rows = {
        pair: {row["command_id"]: row for row in result["points"]}
        for pair, result in pair_results.items()
    }
    points: list[dict[str, Any]] = []
    unresolved = False
    for command_id in selected_commands:
        edge_acceptance: dict[tuple[str, str], bool] = {}
        for pair in combinations(supporting_extractors, 2):
            edge_acceptance[pair] = bool(
                pair_rows[pair].get(command_id, {}).get("accepted") is True
            )
        cliques = _complete_cliques(supporting_extractors, edge_acceptance)
        selected_clique = cliques[0] if len(cliques) == 1 else None
        pair_acceptance = {
            "__".join(pair): edge_acceptance[pair]
            for pair in sorted(edge_acceptance)
        }
        if selected_clique is None:
            unresolved = True
            points.append(
                {
                    "command_id": command_id,
                    "status": "unresolved",
                    "selected_clique": None,
                    "reported_pair": None,
                    "maximum_cliques": [list(clique) for clique in cliques],
                    "measurement": None,
                    "pair_acceptance": pair_acceptance,
                }
            )
            continue
        reported_pair = tuple(sorted(selected_clique[:2]))
        measurement = json.loads(canonical_bytes(pair_rows[reported_pair][command_id]))
        uncertainty = measurement["combined_digitization_uncertainty"]
        if uncertainty is not None:
            between_components = [
                pair_rows[pair][command_id]["combined_digitization_uncertainty"][
                    "between_extractor_component"
                ]
                for pair in combinations(selected_clique, 2)
            ]
            _require(
                all(
                    type(component) in {int, float}
                    and math.isfinite(float(component))
                    and float(component) >= 0
                    for component in between_components
                ),
                "accepted clique has an invalid between-extractor component",
            )
            uncertainty["between_extractor_component"] = float(max(between_components))
        points.append(
            {
                "command_id": command_id,
                "status": "resolved",
                "selected_clique": list(selected_clique),
                "reported_pair": list(reported_pair),
                "maximum_cliques": [list(selected_clique)],
                "measurement": measurement,
                "pair_acceptance": pair_acceptance,
            }
        )
    core = {
        **base,
        "panel_resolution": {
            "status": "resolved_available",
            "selected_status": "available",
            "unavailable_reasons": [],
        },
        "command_set_resolution": {
            "status": "resolved",
            "command_ids": list(selected_commands),
            "supporting_extractor_ids": supporting_extractors,
            "votes": vote_rows,
        },
        "points": points,
        "status": "four_extractions_unresolved" if unresolved else "four_extractions_resolved",
        "unresolved": unresolved,
    }
    core["sha256"] = digest(core)
    return core


def create_output(path: Path, value: Mapping[str, Any]) -> None:
    """Create one canonical output without replacing any existing evidence."""
    target = Path(path)
    _require(target.parent.is_dir(), "output parent does not exist")
    payload = canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PopulationCurveDigitizationError(f"refusing to overwrite existing output: {target}") from exc
    except OSError as exc:
        raise PopulationCurveDigitizationError(f"cannot create output: {target}") from exc


def create_provenance_sidecar(path: Path, *, role: str) -> Path:
    """Bind a manual annotation or NumPy-derived measurement to its device role."""

    supplied = Path(path)
    _require(not supplied.is_symlink(), "provenance artifact must not be a symbolic link")
    artifact = supplied.resolve()
    _require(
        artifact.is_file() and not artifact.is_symlink(),
        "provenance artifact must be a regular file",
    )
    _require(
        role in {"manual_native_pixel_annotation", "numpy_measurement_derivation"},
        "provenance role is invalid",
    )
    sidecar = Path(str(artifact) + ".prov.json")
    core = {
        "schema": PROVENANCE_SCHEMA,
        "scientific_verdict": None,
        "artifact": {"filename": artifact.name, "sha256": _file_digest(artifact)},
        "role": role,
        "device": (
            "human_visual_native_pixels"
            if role == "manual_native_pixel_annotation"
            else "local_cpu"
        ),
        "sim_backend": (
            "not_applicable_manual_annotation"
            if role == "manual_native_pixel_annotation"
            else "numpy"
        ),
    }
    core["sha256"] = digest(core)
    create_output(sidecar, core)
    return sidecar


def _record_from_path(path: Path) -> Mapping[str, Any]:
    return _load_json(Path(path), "extraction record")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path, action="append")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _require(len(args.record) in {1, 2, 3, 4}, "supply one, two blind, three, or four adjudication extractions")
    authority = load_protocol(args.protocol)
    records = [_record_from_path(path) for path in args.record]
    if len(records) == 1:
        output = digitize_record(records[0], authority)
    elif len(records) == 2:
        output = compare_blind_extractions(records[0], records[1], authority)
    elif len(records) == 3:
        output = adjudicate_three_extractions(records[0], records[1], records[2], authority)
    else:
        output = adjudicate_four_extractions(
            records[0], records[1], records[2], records[3], authority
        )
    create_output(args.out, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
