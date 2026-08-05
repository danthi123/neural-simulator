#!/usr/bin/env python3
"""Materialize blind extractor-C native-pixel annotations and records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "research/specs/v14_snr_stageB_population_digitization_protocol_v1.json"
MANIFEST_PATH = ROOT / "research/specs/v14_snr_stageB_primary_figure_asset_manifest_v1.json"
EXTRACTOR_ID = "extractor_c"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(path: Path, value: object) -> None:
    payload = canonical_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def region(x: float, y: float, hx: float = 4.0, hy: float = 4.0) -> dict[str, float]:
    return {
        "x_min": x - hx,
        "x_max": x + hx,
        "y_min": y - hy,
        "y_max": y + hy,
    }


def bounds(x_min: float, y_min: float, x_max: float, y_max: float) -> dict[str, float]:
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def axis(
    scale: str,
    spine: dict[str, float],
    anchors: list[tuple[float, float, float]],
) -> dict[str, object]:
    return {
        "scale": scale,
        "all_legible_major_ticks_included": True,
        "outermost_ticks_included": True,
        "axis_spine_region": spine,
        "anchors": [
            {"pixel_region": region(x, y, 3.0, 3.0), "value": value}
            for x, y, value in anchors
        ],
    }


def bio_unavailable() -> dict[str, str]:
    return {"status": "unavailable"}


def bio_two_sided(x: float, lower_y: float, upper_y: float) -> dict[str, object]:
    return {
        "status": "available",
        "kind": "standard_error",
        "lower_endpoint_region": region(x, lower_y, 4.0, 3.0),
        "upper_endpoint_region": region(x, upper_y, 4.0, 3.0),
    }


def bio_upper(x: float, upper_y: float) -> dict[str, object]:
    return {
        "status": "one_sided",
        "kind": "standard_error",
        "side": "upper",
        "endpoint_region": region(x, upper_y, 4.0, 3.0),
    }


def available_point(
    command_id: str,
    x: float,
    y: float,
    *,
    authoritative_x: float | None,
    bio: dict[str, object] | None = None,
    hx: float = 5.0,
    hy: float = 5.0,
    partial: bool = False,
) -> dict[str, object]:
    return {
        "command_id": command_id,
        "authoritative_x": authoritative_x,
        "status": "available",
        "unavailable_reason": None,
        "marker_center_region": region(x, y, hx, hy),
        "occlusion": {
            "partial": partial,
            "opposing_marker_edges_visible": True,
            "bounded_without_curve_interpolation": True,
            "unique_series_identity": True,
        },
        "biological_error": bio if bio is not None else bio_unavailable(),
    }


def unavailable_point(command_id: str, authoritative_x: float) -> dict[str, object]:
    return {
        "command_id": command_id,
        "authoritative_x": authoritative_x,
        "status": "unavailable",
        "unavailable_reason": "overlap_prevents_unique_series_assignment",
        "marker_center_region": None,
        "occlusion": None,
        "biological_error": None,
    }


def source_points(
    rows: list[tuple[float, float, bool, dict[str, object] | None, float, float]],
) -> list[dict[str, object]]:
    return [
        available_point(
            f"command_{index:03d}",
            x,
            y,
            authoritative_x=None,
            bio=bio,
            hx=hx,
            hy=hy,
            partial=partial,
        )
        for index, (x, y, partial, bio, hx, hy) in enumerate(rows, start=1)
    ]


def published_points(
    rows: list[tuple[float, float, float, bool, dict[str, object] | None]],
) -> list[dict[str, object]]:
    return [
        available_point(
            f"command_{index:03d}",
            x,
            y,
            authoritative_x=authoritative_x,
            bio=bio,
            partial=partial,
        )
        for index, (authoritative_x, x, y, partial, bio) in enumerate(rows, start=1)
    ]


def build_panel_payloads() -> dict[tuple[str, str], dict[str, object]]:
    a4_points: list[dict[str, object]] = [
        unavailable_point("command_001", -60),
        unavailable_point("command_002", -55),
        unavailable_point("command_003", -50),
        available_point("command_004", 853, 964 + 4500, authoritative_x=-45, bio=bio_two_sided(853, 1004 + 4500, 948 + 4500)),
        available_point("command_005", 957, 876 + 4500, authoritative_x=-40, bio=bio_upper(957, 865 + 4500)),
        available_point("command_006", 1061, 760 + 4500, authoritative_x=-35, bio=bio_upper(1061, 730 + 4500)),
        available_point("command_007", 1166, 567 + 4500, authoritative_x=-30, bio=bio_two_sided(1166, 608 + 4500, 530 + 4500)),
        available_point("command_008", 1270, 382 + 4500, authoritative_x=-25, partial=True),
        available_point("command_009", 1375, 258 + 4500, authoritative_x=-20, bio=bio_two_sided(1375, 286 + 4500, 230 + 4500)),
        available_point("command_010", 1479, 183 + 4500, authoritative_x=-15, bio=bio_two_sided(1479, 212 + 4500, 156 + 4500)),
        available_point("command_011", 1584, 108 + 4500, authoritative_x=-10, bio=bio_upper(1584, 78 + 4500)),
        available_point("command_012", 1688, 97 + 4500, authoritative_x=-5),
        available_point("command_013", 1792, 127 + 4500, authoritative_x=0, bio=bio_upper(1792, 91 + 4500)),
        unavailable_point("command_014", 5),
    ]

    b4_rows = [
        (3389, 4747, True, None, 6.0, 6.0),
        (3463, 4725, True, None, 6.0, 6.0),
        (3611, 4779, False, None, 5.0, 5.0),
        (3683, 4810, False, None, 5.0, 5.0),
        (3755, 4847, False, None, 5.0, 5.0),
        (3830, 4941, False, None, 5.0, 5.0),
        (3897, 5054, True, None, 9.0, 6.0),
        (3977, 5161, False, None, 5.0, 5.0),
        (4051, 5304, False, None, 5.0, 5.0),
        (4129, 5394, False, None, 6.0, 5.0),
        (4192, 5447, False, None, 6.0, 5.0),
        (4340, 5492, True, None, 6.0, 5.0),
        (4409, 5500, True, None, 6.0, 5.0),
        (4491, 5527, True, None, 6.0, 5.0),
    ]

    recovery_rows = [
        (972, 5755, True, None, 8.0, 8.0),
        (988, 5682, True, None, 8.0, 8.0),
        (1004, 5647, True, None, 8.0, 8.0),
        (1032, 5560, True, None, 8.0, 8.0),
        (1049, 5540, True, None, 8.0, 8.0),
        (1079, 5477, True, None, 8.0, 8.0),
        (1108, 5500, True, None, 8.0, 8.0),
        (1249, 5340, False, bio_two_sided(1249, 5381, 5303), 5.0, 5.0),
        (1389, 5316, False, bio_two_sided(1389, 5368, 5269), 5.0, 5.0),
        (1531, 5240, False, bio_two_sided(1531, 5296, 5188), 5.0, 5.0),
        (1669, 5257, False, None, 5.0, 5.0),
        (1810, 5212, False, bio_two_sided(1810, 5261, 5167), 5.0, 5.0),
        (1949, 5266, False, None, 5.0, 5.0),
        (2090, 5110, False, bio_two_sided(2090, 5168, 5056), 5.0, 6.0),
        (2231, 5223, False, None, 5.0, 5.0),
        (2372, 5216, False, bio_upper(2372, 5150), 5.0, 5.0),
        (2513, 5223, False, bio_upper(2513, 5188), 6.0, 5.0),
    ]

    sodium_deactivation_rows = [
        (-100, 699, 4052, False, None),
        (-90, 836, 4058, False, None),
        (-80, 971, 4054, False, None),
        (-70, 1109, 4042, False, None),
        (-60, 1246, 4022, False, None),
        (-50, 1383, 3976, False, None),
        (-40, 1518, 3878, False, bio_two_sided(1518, 3911, 3848)),
        (-30, 1654, 3638, False, bio_two_sided(1654, 3680, 3605)),
        (-20, 1791, 3544, False, None),
    ]

    kv3_activation_rows = [
        (-80, 322, 2706, False, None),
        (-70, 374, 2706, False, None),
        (-60, 424, 2706, False, None),
        (-50, 474, 2701, False, None),
        (-40, 526, 2697, False, None),
        (-30, 576, 2678, False, None),
        (-20, 626, 2612, False, None),
        (-10, 675, 2508, False, None),
        (0, 727, 2417, False, bio_two_sided(727, 2440, 2396)),
        (10, 777, 2361, False, bio_two_sided(777, 2388, 2337)),
        (20, 827, 2328, False, bio_two_sided(827, 2375, 2301)),
        (30, 879, 2320, False, bio_two_sided(879, 2354, 2292)),
        (40, 930, 2322, False, bio_two_sided(930, 2354, 2298)),
        (50, 979, 2317, False, None),
    ]

    kv3_inactivation_rows = [
        (-110, 1534, 2315, False, None),
        (-100, 1593, 2315, False, None),
        (-90, 1651, 2315, False, None),
        (-80, 1710, 2320, False, bio_two_sided(1710, 2345, 2303)),
        (-70, 1767, 2344, False, bio_two_sided(1767, 2371, 2318)),
        (-60, 1825, 2397, False, bio_two_sided(1825, 2421, 2377)),
        (-50, 1884, 2493, False, bio_two_sided(1884, 2521, 2470)),
        (-40, 1943, 2594, False, None),
        (-30, 2001, 2660, False, bio_two_sided(2001, 2677, 2639)),
        (-20, 2057, 2686, False, None),
        (-10, 2116, 2686, False, None),
        (0, 2174, 2686, False, None),
    ]

    kv3_deactivation_rows = [
        (-70, 2252, 777, False, bio_two_sided(2252, 800, 759)),
        (-60, 2352, 661, False, bio_two_sided(2352, 683, 643)),
        (-50, 2450, 528, False, bio_two_sided(2450, 555, 506)),
        (-40, 2549, 441, False, bio_two_sided(2549, 467, 420)),
        (-30, 2649, 275, False, bio_two_sided(2649, 299, 254)),
    ]

    return {
        ("ding-wei-zhou-2011-figure-6", "A4"): {
            "slug": "fast_na_activation",
            "x_axis": axis(
                "linear",
                bounds(539, 5596, 1935, 5604),
                [(539, 5600, -60), (957, 5600, -40), (1375, 5600, -20), (1792, 5600, 0)],
            ),
            "y_axis": axis(
                "linear",
                bounds(535, 4583, 543, 5555),
                [(539, 5552, 0), (539, 5069, 0.5), (539, 4586, 1.0)],
            ),
            "points": a4_points,
            "annotation_notes": [
                "Filled-circle SNr GABA markers at -60, -55, -50, and +5 mV cannot be uniquely separated from the open-circle series.",
                "The -25 mV marker center is bounded from opposing visible edges despite arrow overlap.",
            ],
        },
        ("ding-wei-zhou-2011-figure-6", "B4"): {
            "slug": "fast_na_steady_state_inactivation",
            "x_axis": axis(
                "linear",
                bounds(3216, 5583, 4570, 5591),
                [(3389, 5587, -100), (3686, 5587, -80), (3977, 5587, -60), (4268, 5587, -40), (4564, 5587, -20)],
            ),
            "y_axis": axis(
                "linear",
                bounds(3212, 4695, 3220, 5555),
                [(3216, 5550, 0), (3216, 5122, 0.5), (3216, 4700, 1.0)],
            ),
            "points": source_points(b4_rows),
            "annotation_notes": [
                "Source-marker x coordinates are retained without forcing a command ladder.",
                "Three crowded filled markers whose centers were not bounded independently are omitted; visible means remain measurable at the retained markers.",
                "SEM endpoints are unavailable where open-series bars, curves, or arrows prevent endpoint-specific assignment.",
            ],
        },
        ("ding-wei-zhou-2011-figure-7", "D"): {
            "slug": "fast_na_recovery",
            "x_axis": axis(
                "linear",
                bounds(967, 6201, 2600, 6209),
                [(967, 6205, 0), (1250, 6205, 40), (1530, 6205, 80), (1808, 6205, 120), (2090, 6205, 160), (2372, 6205, 200)],
            ),
            "y_axis": axis(
                "linear",
                bounds(894, 5194, 902, 6159),
                [(898, 6155, 0), (898, 5676, 0.5), (898, 5199, 1.0)],
            ),
            "points": source_points(recovery_rows),
            "annotation_notes": [
                "Source-marker x coordinates are retained without imposing the operational recovery ladder.",
                "Early filled markers are partially occluded by adjacent filled markers but have bounded opposing edges; their SEM endpoints are not separately distinguishable.",
            ],
        },
        ("ding-wei-zhou-2011-figure-9", "C"): {
            "slug": "fast_na_deactivation",
            "x_axis": axis(
                "linear",
                bounds(836, 4091, 1656, 4099),
                [(836, 4095, -90), (1248, 4095, -60), (1656, 4095, -30)],
            ),
            "y_axis": axis(
                "linear",
                bounds(558, 3185, 566, 3876),
                [(562, 3872, 0.1), (562, 3531, 0.2), (562, 3189, 0.3)],
            ),
            "points": published_points(sodium_deactivation_rows),
            "annotation_notes": [
                "The calibration spine regions cover the full interval between the outermost printed major labels; unlabeled minor ticks are not used as anchors.",
                "Only the clearly separated -40 and -30 mV filled-series SEM endpoints are bounded.",
            ],
        },
        ("ding-matta-zhou-2011-figure-8", "C1"): {
            "slug": "kv3_activation",
            "x_axis": axis(
                "linear",
                bounds(272, 2740, 1031, 2748),
                [(272, 2744, -90), (424, 2744, -60), (576, 2744, -30), (728, 2744, 0), (879, 2744, 30), (1030, 2744, 60)],
            ),
            "y_axis": axis(
                "linear",
                bounds(268, 2314, 276, 2709),
                [(272, 2706, 0), (272, 2511, 0.5), (272, 2318, 1.0)],
            ),
            "points": published_points(kv3_activation_rows),
            "annotation_notes": [
                "Every published command marker is bounded; endpoints are retained only where the vertical SEM extent is distinct from the marker and connecting line.",
            ],
        },
        ("ding-matta-zhou-2011-figure-8", "C2"): {
            "slug": "kv3_steady_state_inactivation",
            "x_axis": axis(
                "linear",
                bounds(1476, 2740, 2176, 2748),
                [(1476, 2744, -120), (1652, 2744, -90), (1826, 2744, -60), (2002, 2744, -30), (2174, 2744, 0)],
            ),
            "y_axis": axis(
                "linear",
                bounds(1472, 2312, 1480, 2707),
                [(1476, 2704, 0), (1476, 2511, 0.5), (1476, 2316, 1.0)],
            ),
            "points": published_points(kv3_inactivation_rows),
            "annotation_notes": [
                "Every published command marker is bounded; crowded or marker-limited SEM extents remain unavailable.",
            ],
        },
        ("ding-matta-zhou-2011-figure-9", "B"): {
            "slug": "kv3_deactivation",
            "x_axis": axis(
                "linear",
                bounds(2152, 804, 2747, 812),
                [(2152, 808, -80), (2352, 808, -60), (2550, 808, -40), (2747, 808, -20)],
            ),
            "y_axis": axis(
                "log10",
                bounds(2148, 236, 2156, 612),
                [(2152, 609, 1), (2152, 425, 2), (2152, 240, 4)],
            ),
            "points": published_points(kv3_deactivation_rows),
            "annotation_notes": [
                "All five filled-circle means and both visible endpoints of each standard-error bar are bounded on the printed log10 y axis.",
            ],
        },
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol_binding = {
        "path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "sha256": file_sha256(PROTOCOL_PATH),
    }
    assets = {row["id"]: row for row in manifest["assets"]}
    panels = {(row["asset_id"], row["panel"]): row for row in protocol["eligible_panels"]}
    payloads = build_panel_payloads()
    if set(payloads) != set(panels):
        raise RuntimeError("extractor-C payloads do not cover the frozen eligible-panel set")

    targets: list[tuple[Path, object]] = []
    for key, payload in payloads.items():
        asset_id, panel_id = key
        panel = panels[key]
        asset = assets[asset_id]
        acquisition = asset["full_resolution_acquisition"]
        slug = str(payload["slug"])
        workflow_id = f"workflow_c_{slug}"
        annotation_path = OUT / "annotations" / f"{slug}.annotation.json"
        record_path = OUT / "records" / f"{slug}.record.json"
        image_path = ROOT / acquisition["local_image_path"]
        receipt_path = ROOT / acquisition["local_receipt_path"]
        panel_bounds = panel["panel_bounds_original_pixels"]
        normalized_bounds = bounds(panel_bounds[0], panel_bounds[1], panel_bounds[2], panel_bounds[3])
        measurement = {
            "x_axis": payload["x_axis"],
            "y_axis": payload["y_axis"],
            "points": payload["points"],
        }
        annotation = {
            "schema": "v14-snr-stageB-population-native-pixel-annotation-v1",
            "annotation_id": f"annotation_c_{slug}",
            "workflow_id": workflow_id,
            "extractor_id": EXTRACTOR_ID,
            "blindness_attestation": {
                "blind_to_other_extractions": True,
                "prohibited_paths_accessed": False,
                "statement": "Extractor C was performed without listing, reading, searching, hashing, or inspecting extractor A, extractor B, or comparison artifacts.",
            },
            "protocol": protocol_binding,
            "asset": {
                "asset_id": asset_id,
                "image_path": acquisition["local_image_path"],
                "image_sha256": acquisition["assembled_image_sha256"],
                "pixel_sha256": acquisition["pixel_sha256"],
                "width": asset["full_resolution_pixels"][0],
                "height": asset["full_resolution_pixels"][1],
            },
            "panel": {
                "id": panel_id,
                "series_id": panel["series_identity"],
                "bounds_original_pixels": normalized_bounds,
            },
            "coordinate_space": {
                "origin": "top_left_of_full_resolution_source_image",
                "units": "original_image_pixels",
                "source_pixels_resampled": False,
                "display_zoom": "nearest_neighbor_only",
                "pixel_regions": "continuous_uniform_center_uncertainty_rectangles",
            },
            "annotation_tool": {
                "name": "native_original_pixel_annotation",
                "version": "workflow_c_1",
            },
            "inspection_method": {
                "source": "official full-resolution PNG bound by manifest and tile receipt",
                "native_crop": "ImageMagick pixel-preserving crop only",
                "zoom": "nearest-neighbor only",
                "center_aid": "native-pixel threshold and distance-transform checks followed by visual bounding",
                "curve_interpolation_used": False,
            },
            "measurement": measurement,
            "notes": payload["annotation_notes"],
            "status": "available",
        }
        annotation_payload = canonical_bytes(annotation) + b"\n"
        annotation_digest = hashlib.sha256(annotation_payload).hexdigest()
        record = {
            "schema": "v14-snr-stageB-population-curve-extraction-v1",
            "record_id": f"record_c_{slug}",
            "extractor_id": EXTRACTOR_ID,
            "protocol": protocol_binding,
            "provenance": {
                "workflow_id": workflow_id,
                "blind_to_other_extractions": True,
                "source_pixels_resampled": False,
                "tool": {
                    "name": "native_original_pixel_annotation",
                    "version": "workflow_c_1",
                },
                "annotation_artifact": {
                    "path": str(annotation_path.relative_to(ROOT)),
                    "sha256": annotation_digest,
                },
            },
            "asset": {
                "asset_id": asset_id,
                "receipt": {
                    "path": acquisition["local_receipt_path"],
                    "file_sha256": acquisition["receipt_file_sha256"],
                    "self_sha256": acquisition["receipt_self_sha256"],
                },
                "image": {
                    "path": acquisition["local_image_path"],
                    "sha256": acquisition["assembled_image_sha256"],
                    "pixel_sha256": acquisition["pixel_sha256"],
                },
            },
            "panel": {
                "id": panel_id,
                "series_id": panel["series_identity"],
                "bounds": normalized_bounds,
            },
            "status": "available",
            "unavailable_reason": None,
            "measurement": measurement,
        }
        if file_sha256(image_path) != acquisition["assembled_image_sha256"]:
            raise RuntimeError(f"image digest mismatch before writing {slug}")
        if file_sha256(receipt_path) != acquisition["receipt_file_sha256"]:
            raise RuntimeError(f"receipt digest mismatch before writing {slug}")
        if annotation_path.exists():
            if annotation_path.read_bytes() != annotation_payload:
                raise RuntimeError(f"existing annotation differs from extractor-C payload: {slug}")
        else:
            targets.append((annotation_path, annotation))
        targets.append((record_path, record))

    existing = [str(path) for path, _ in targets if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite extractor-C evidence: {existing}")
    for path, value in targets:
        create(path, value)


if __name__ == "__main__":
    main()
