#!/usr/bin/env python3
"""Acquire an exact official PMC tileshop view and emit a custody receipt."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


SCHEMA = "sim-pmc-tile-asset-receipt-v1"
_HOST = "www.ncbi.nlm.nih.gov"
_PATH = "/corecgi/tileshop/tileshop.fcgi"
_MANIFEST_TYPES = {"application/javascript", "text/javascript", "text/plain"}
_IMAGE_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_DIGITS = re.compile(r"[0-9]{1,20}")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_DIMENSION = 100_000
_MAX_TILE_DIMENSION = 4_096
_MAX_TILES = 100_000
_MAX_IMAGE_PIXELS = 500_000_000


class PmcTileAssetError(ValueError):
    """Raised when an asset cannot be acquired without ambiguity."""


@dataclass(frozen=True)
class FetchResponse:
    """A complete fetch result used by both HTTP and offline test fetchers."""

    body: bytes
    content_type: str
    final_url: str
    status: int = 200


Fetcher = Callable[[str], FetchResponse]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PmcTileAssetError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PmcTileAssetError("receipt contains a non-canonical value") from exc


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("sha256", None)
    return _digest_bytes(_canonical_bytes(body))


class _ManifestParser:
    """Parser for the restricted JavaScript object literal emitted by tileshop."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> dict[str, Any]:
        value = self._value(0)
        self._space()
        _require(self.index == len(self.text), "manifest has trailing content")
        _require(isinstance(value, dict), "manifest root must be an object")
        return value

    def _space(self) -> None:
        while self.index < len(self.text) and self.text[self.index] in " \t\r\n":
            self.index += 1

    def _take(self, character: str) -> None:
        self._space()
        _require(
            self.index < len(self.text) and self.text[self.index] == character,
            f"manifest expected {character!r} at byte {self.index}",
        )
        self.index += 1

    def _value(self, depth: int) -> Any:
        _require(depth <= 8, "manifest nesting is too deep")
        self._space()
        _require(self.index < len(self.text), "manifest ended unexpectedly")
        character = self.text[self.index]
        if character == "{":
            return self._object(depth + 1)
        if character == "[":
            return self._array(depth + 1)
        if character == '"':
            return self._string()
        match = _NUMBER.match(self.text, self.index)
        if match is not None:
            token = match.group(0)
            self.index = match.end()
            value = float(token) if any(c in token for c in ".eE") else int(token)
            _require(not isinstance(value, float) or math.isfinite(value), "manifest number is not finite")
            return value
        raise PmcTileAssetError(f"manifest value is invalid at byte {self.index}")

    def _key(self) -> str:
        self._space()
        if self.index < len(self.text) and self.text[self.index] == '"':
            return self._string()
        match = _IDENTIFIER.match(self.text, self.index)
        _require(match is not None, f"manifest key is invalid at byte {self.index}")
        self.index = match.end()
        return match.group(0)

    def _string(self) -> str:
        start = self.index
        self.index += 1
        escaped = False
        while self.index < len(self.text):
            character = self.text[self.index]
            if ord(character) < 0x20:
                raise PmcTileAssetError("manifest string contains a control character")
            self.index += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                token = self.text[start:self.index]
                try:
                    value = json.loads(token)
                except json.JSONDecodeError as exc:
                    raise PmcTileAssetError("manifest string escape is invalid") from exc
                _require(isinstance(value, str), "manifest string is invalid")
                return value
        raise PmcTileAssetError("manifest string is unterminated")

    def _object(self, depth: int) -> dict[str, Any]:
        self._take("{")
        result: dict[str, Any] = {}
        self._space()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
            return result
        while True:
            key = self._key()
            _require(key not in result, f"manifest has duplicate key {key!r}")
            self._take(":")
            result[key] = self._value(depth)
            self._space()
            _require(self.index < len(self.text), "manifest object is unterminated")
            if self.text[self.index] == "}":
                self.index += 1
                return result
            self._take(",")

    def _array(self, depth: int) -> list[Any]:
        self._take("[")
        result: list[Any] = []
        self._space()
        if self.index < len(self.text) and self.text[self.index] == "]":
            self.index += 1
            return result
        while True:
            result.append(self._value(depth))
            _require(len(result) <= _MAX_TILES, "manifest array is too large")
            self._space()
            _require(self.index < len(self.text), "manifest array is unterminated")
            if self.text[self.index] == "]":
                self.index += 1
                return result
            self._take(",")


def parse_manifest(text: str) -> dict[str, Any]:
    """Parse and structurally validate a tileshop manifest without evaluation."""
    _require(isinstance(text, str), "manifest text must be a string")
    _require(len(text.encode("utf-8")) <= _MAX_MANIFEST_BYTES, "manifest is too large")
    manifest = _ManifestParser(text).parse()
    _require(
        set(manifest) == {"ProjectName", "ImageName", "Header", "Footer", "Sat", "aView"},
        "manifest root fields are invalid",
    )
    _require(
        isinstance(manifest["ProjectName"], str)
        and _SAFE_NAME.fullmatch(manifest["ProjectName"]) is not None,
        "manifest project name is unsafe",
    )
    _require(
        isinstance(manifest["ImageName"], str)
        and _SAFE_NAME.fullmatch(manifest["ImageName"]) is not None
        and ".." not in manifest["ImageName"],
        "manifest image name is unsafe",
    )
    _require(manifest["Header"] == "" and manifest["Footer"] == "", "manifest decorations are unsupported")
    _require(
        isinstance(manifest["Sat"], str) and _DIGITS.fullmatch(manifest["Sat"]) is not None,
        "manifest satellite is invalid",
    )
    views = manifest["aView"]
    _require(isinstance(views, list) and views, "manifest views must be a nonempty array")
    seen: set[str] = set()
    for view in views:
        _require(isinstance(view, dict), "manifest view must be an object")
        _require(
            set(view) == {"sId", "sName", "fScale", "W", "H", "w", "h"},
            "manifest view fields are invalid",
        )
        view_id = view["sId"]
        _require(isinstance(view_id, str) and _DIGITS.fullmatch(view_id) is not None, "view id is invalid")
        _require(view_id not in seen, f"manifest has duplicate view id {view_id!r}")
        seen.add(view_id)
        _require(
            isinstance(view["sName"], str)
            and 0 < len(view["sName"]) <= 64
            and all(ord(c) >= 0x20 for c in view["sName"]),
            "view name is invalid",
        )
        _require(
            type(view["fScale"]) in {int, float}
            and math.isfinite(float(view["fScale"]))
            and float(view["fScale"]) > 0,
            "view scale is invalid",
        )
        for field in ("W", "H"):
            _require(type(view[field]) is int and 0 < view[field] <= _MAX_DIMENSION, f"view {field} is invalid")
        for field in ("w", "h"):
            _require(
                type(view[field]) is int and 0 < view[field] <= _MAX_TILE_DIMENSION,
                f"view {field} is invalid",
            )
        rows = math.ceil(view["H"] / view["h"])
        columns = math.ceil(view["W"] / view["w"])
        _require(rows * columns <= _MAX_TILES, "view tile grid is too large")
        _require(view["W"] * view["H"] <= _MAX_IMAGE_PIXELS, "view image is too large")
    return manifest


def _content_type(value: str) -> str:
    _require(isinstance(value, str), "response content type is missing")
    return value.split(";", 1)[0].strip().lower()


def _fetch(fetcher: Fetcher, url: str, allowed_types: set[str], context: str) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = fetcher(url)
            _require(isinstance(response, FetchResponse), f"{context} fetcher returned an invalid response")
            _require(response.status == 200, f"{context} returned HTTP status {response.status}")
            _require(response.final_url == url, f"{context} redirected or changed URL")
            _require(type(response.body) is bytes and response.body, f"{context} is missing")
            content_type = _content_type(response.content_type)
            _require(content_type in allowed_types, f"{context} content type is invalid: {content_type}")
            return response.body, content_type
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.1 * (2**attempt))
    if isinstance(last_error, PmcTileAssetError):
        raise last_error
    raise PmcTileAssetError(f"{context} fetch failed after 3 attempts") from last_error


def http_fetch(url: str) -> FetchResponse:
    """Fetch one URL with the standard library and a bounded timeout."""
    request = Request(
        url,
        headers={
            "Accept": "image/jpeg,image/png,text/plain,*/*;q=0.1",
            "User-Agent": "sim-pmc-tile-asset/1",
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - URL is validated before use.
        body = response.read()
        return FetchResponse(
            body=body,
            content_type=response.headers.get("Content-Type", ""),
            final_url=response.geturl(),
            status=response.status,
        )


def _manifest_url(value: str) -> tuple[str, str]:
    _require(isinstance(value, str), "manifest URL must be a string")
    parsed = urlsplit(value)
    _require(
        parsed.scheme == "https"
        and parsed.hostname == _HOST
        and parsed.netloc == _HOST
        and parsed.path == _PATH
        and not parsed.fragment,
        "manifest URL is not the official PMC tileshop endpoint",
    )
    try:
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise PmcTileAssetError("manifest URL query is malformed") from exc
    query: dict[str, str] = {}
    for key, item in pairs:
        _require(key not in query, f"manifest URL has duplicate query key {key!r}")
        query[key] = item
    _require(set(query) == {"manifest", "p", "id", "w", "h"}, "manifest URL query fields are invalid")
    _require(query["manifest"] == "1", "manifest URL does not request a manifest")
    _require(_SAFE_NAME.fullmatch(query["p"]) is not None, "manifest URL project is unsafe")
    _require(
        _SAFE_NAME.fullmatch(query["id"]) is not None and ".." not in query["id"],
        "manifest URL image name is unsafe",
    )
    for field in ("w", "h"):
        _require(
            _DIGITS.fullmatch(query[field]) is not None and int(query[field]) > 0,
            f"manifest URL {field} is invalid",
        )
    return query["p"], query["id"]


def _tile_url(project: str, view_id: str, satellite: str, row: int, column: int) -> str:
    query = urlencode(
        [("p", project), ("id", view_id), ("s", satellite), ("r", str(row)), ("c", str(column))]
    )
    return urlunsplit(("https", _HOST, _PATH, query, ""))


def _load_expected(value: Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        document = dict(value)
    else:
        try:
            document = json.loads(Path(value).read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PmcTileAssetError("expected receipt is not valid JSON") from exc
    _require(isinstance(document, dict), "expected receipt must be an object")
    _require(document.get("schema") == SCHEMA, "expected receipt schema is invalid")
    _require(document.get("sha256") == _self_digest(document), "expected receipt self digest is invalid")
    return document


def _create_path(path: Path, payload: bytes, context: str) -> None:
    _require(path.parent.is_dir(), f"{context} parent does not exist")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(path, flags, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PmcTileAssetError(f"refusing to overwrite existing {context}: {path}") from exc
    except OSError as exc:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise PmcTileAssetError(f"cannot create {context}: {path}") from exc


def _tile_image(
    body: bytes,
    content_type: str,
    allowed_sizes: set[tuple[int, int]],
    context: str,
) -> Image.Image:
    try:
        with Image.open(BytesIO(body)) as probe:
            probe.verify()
        with Image.open(BytesIO(body)) as source:
            source.load()
            _require(source.format == _IMAGE_TYPES[content_type], f"{context} format contradicts content type")
            _require(source.size in allowed_sizes, f"{context} dimensions are invalid")
            _require(source.mode in {"L", "RGB", "RGBA"}, f"{context} mode is unsupported")
            return source.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise PmcTileAssetError(f"{context} content is not a valid image") from exc


def acquire(
    manifest_url: str,
    view_id: str,
    output_path: Path,
    receipt_path: Path,
    *,
    fetcher: Fetcher = http_fetch,
    expected_receipt: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Acquire one exact manifest view, writing a PNG and receipt create-only."""
    project, image_name = _manifest_url(manifest_url)
    _require(isinstance(view_id, str) and _DIGITS.fullmatch(view_id) is not None, "requested view id is invalid")
    output_path = Path(output_path)
    receipt_path = Path(receipt_path)
    _require(output_path.suffix.lower() == ".png", "assembled output must use a .png suffix")
    _require(output_path.absolute() != receipt_path.absolute(), "output and receipt paths must differ")
    _require(not os.path.lexists(output_path), f"refusing to overwrite existing output: {output_path}")
    _require(not os.path.lexists(receipt_path), f"refusing to overwrite existing receipt: {receipt_path}")
    expected = _load_expected(expected_receipt)

    manifest_body, manifest_type = _fetch(fetcher, manifest_url, _MANIFEST_TYPES, "manifest")
    _require(len(manifest_body) <= _MAX_MANIFEST_BYTES, "manifest is too large")
    try:
        manifest_text = manifest_body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PmcTileAssetError("manifest is not UTF-8") from exc
    manifest = parse_manifest(manifest_text)
    _require(manifest["ProjectName"] == project, "manifest project does not match its URL")
    _require(manifest["ImageName"] == image_name, "manifest image does not match its URL")
    matches = [item for item in manifest["aView"] if item["sId"] == view_id]
    _require(len(matches) == 1, f"requested view {view_id!r} is not declared exactly once")
    view = matches[0]
    rows = math.ceil(view["H"] / view["h"])
    columns = math.ceil(view["W"] / view["w"])
    manifest_hash = _digest_bytes(manifest_body)

    if expected is not None:
        _require(expected.get("manifest", {}).get("url") == manifest_url, "expected manifest URL differs")
        _require(expected.get("manifest", {}).get("sha256") == manifest_hash, "manifest changed from expected receipt")
        _require(expected.get("view", {}).get("sId") == view_id, "expected receipt view differs")
        expected_tiles = expected.get("tiles")
        _require(
            isinstance(expected_tiles, list) and len(expected_tiles) == rows * columns,
            "expected tile grid differs",
        )
    else:
        expected_tiles = None

    canvas: Image.Image | None = None
    tile_records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    tile_index = 0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            url = _tile_url(project, view_id, manifest["Sat"], row, column)
            _require(url not in seen_urls, "tile grid produced a duplicate URL")
            seen_urls.add(url)
            body, content_type = _fetch(fetcher, url, set(_IMAGE_TYPES), f"tile {row},{column}")
            tile_hash = _digest_bytes(body)
            if expected_tiles is not None:
                pinned = expected_tiles[tile_index]
                _require(
                    isinstance(pinned, dict)
                    and pinned.get("row") == row
                    and pinned.get("column") == column
                    and pinned.get("url") == url,
                    f"expected tile order changed at {row},{column}",
                )
                _require(pinned.get("sha256") == tile_hash, f"tile {row},{column} changed from expected receipt")
            remaining_width = view["W"] - (column - 1) * view["w"]
            remaining_height = view["H"] - (row - 1) * view["h"]
            allowed_widths = {view["w"]}
            allowed_heights = {view["h"]}
            if column == columns:
                allowed_widths.add(remaining_width)
            if row == rows:
                allowed_heights.add(remaining_height)
            allowed_sizes = {
                (allowed_width, allowed_height)
                for allowed_width in allowed_widths
                for allowed_height in allowed_heights
            }
            tile = _tile_image(body, content_type, allowed_sizes, f"tile {row},{column}")
            if canvas is None:
                canvas = Image.new(tile.mode, (columns * view["w"], rows * view["h"]))
            _require(tile.mode == canvas.mode, f"tile {row},{column} mode differs from the grid")
            canvas.paste(tile, ((column - 1) * view["w"], (row - 1) * view["h"]))
            tile_records.append(
                {
                    "byte_count": len(body),
                    "column": column,
                    "content_type": content_type,
                    "format": _IMAGE_TYPES[content_type],
                    "height": tile.height,
                    "row": row,
                    "sha256": tile_hash,
                    "url": url,
                    "width": tile.width,
                }
            )
            tile_index += 1
            tile.close()

    _require(canvas is not None, "tile grid is empty")
    cropped = canvas.crop((0, 0, view["W"], view["H"]))
    canvas.close()
    encoded = BytesIO()
    cropped.save(encoded, format="PNG", optimize=False, compress_level=9)
    output_bytes = encoded.getvalue()
    pixel_hash = _digest_bytes(cropped.tobytes())
    mode = cropped.mode
    cropped.close()

    # A cheap second manifest fetch detects source metadata mutation during a long grid download.
    final_manifest_body, final_manifest_type = _fetch(fetcher, manifest_url, _MANIFEST_TYPES, "manifest recheck")
    _require(final_manifest_type == manifest_type, "manifest content type changed during acquisition")
    _require(final_manifest_body == manifest_body, "manifest changed during acquisition")

    output_record = {
        "byte_count": len(output_bytes),
        "content_type": "image/png",
        "format": "PNG",
        "height": view["H"],
        "mode": mode,
        "path": str(output_path),
        "pixel_sha256": pixel_hash,
        "sha256": _digest_bytes(output_bytes),
        "width": view["W"],
    }
    if expected is not None:
        _require(
            expected.get("assembled_image", {}).get("sha256") == output_record["sha256"],
            "assembled image changed from expected receipt",
        )

    receipt: dict[str, Any] = {
        "assembled_image": output_record,
        "grid": {
            "column_count": columns,
            "coordinate_base": 1,
            "row_count": rows,
            "tile_count": rows * columns,
        },
        "manifest": {
            "byte_count": len(manifest_body),
            "content_type": manifest_type,
            "sha256": manifest_hash,
            "text_sha256": _digest_bytes(manifest_text.encode("utf-8")),
            "url": manifest_url,
        },
        "schema": SCHEMA,
        "scientific_verdict": None,
        "source": {
            "image_name": image_name,
            "project": project,
            "satellite": manifest["Sat"],
        },
        "status": "acquired",
        "tiles": tile_records,
        "view": dict(view),
    }
    receipt["sha256"] = _self_digest(receipt)
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii")
        + b"\n"
    )

    _create_path(output_path, output_bytes, "assembled output")
    try:
        _create_path(receipt_path, receipt_bytes, "receipt")
    except Exception:
        try:
            output_path.unlink()
        except OSError:
            pass
        raise
    return receipt


def verify_receipt(path: Path, image_path: Path | None = None) -> dict[str, Any]:
    """Verify a receipt plus the assembled image it authenticates."""
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PmcTileAssetError("receipt is not valid JSON") from exc
    _require(isinstance(value, dict), "receipt must be an object")
    _require(
        set(value)
        == {
            "assembled_image",
            "grid",
            "manifest",
            "schema",
            "scientific_verdict",
            "sha256",
            "source",
            "status",
            "tiles",
            "view",
        },
        "receipt fields are invalid",
    )
    _require(value.get("schema") == SCHEMA, "receipt schema is invalid")
    _require(
        isinstance(value.get("sha256"), str) and _SHA256.fullmatch(value["sha256"]) is not None,
        "receipt digest is invalid",
    )
    _require(value["sha256"] == _self_digest(value), "receipt self digest is invalid")
    _require(value.get("scientific_verdict") is None, "receipt must not contain a scientific verdict")
    _require(value.get("status") == "acquired", "receipt status is invalid")
    assembled = value.get("assembled_image")
    _require(
        isinstance(assembled, dict)
        and set(assembled)
        == {
            "byte_count",
            "content_type",
            "format",
            "height",
            "mode",
            "path",
            "pixel_sha256",
            "sha256",
            "width",
        },
        "receipt assembled image record is invalid",
    )
    _require(
        assembled.get("content_type") == "image/png"
        and assembled.get("format") == "PNG"
        and assembled.get("mode") in {"L", "RGB", "RGBA"}
        and type(assembled.get("byte_count")) is int
        and assembled["byte_count"] > 0
        and type(assembled.get("width")) is int
        and type(assembled.get("height")) is int
        and assembled["width"] > 0
        and assembled["height"] > 0
        and isinstance(assembled.get("sha256"), str)
        and _SHA256.fullmatch(assembled["sha256"]) is not None
        and isinstance(assembled.get("pixel_sha256"), str)
        and _SHA256.fullmatch(assembled["pixel_sha256"]) is not None,
        "receipt assembled image metadata is invalid",
    )
    manifest = value.get("manifest")
    _require(
        isinstance(manifest, dict)
        and set(manifest) == {"byte_count", "content_type", "sha256", "text_sha256", "url"},
        "receipt manifest record is invalid",
    )
    _require(
        type(manifest.get("byte_count")) is int
        and manifest["byte_count"] > 0
        and manifest.get("content_type") in _MANIFEST_TYPES
        and isinstance(manifest.get("sha256"), str)
        and _SHA256.fullmatch(manifest["sha256"]) is not None
        and isinstance(manifest.get("text_sha256"), str)
        and _SHA256.fullmatch(manifest["text_sha256"]) is not None,
        "receipt manifest metadata is invalid",
    )
    project, image_name = _manifest_url(manifest.get("url"))
    source = value.get("source")
    _require(
        isinstance(source, dict)
        and set(source) == {"image_name", "project", "satellite"}
        and source.get("project") == project
        and source.get("image_name") == image_name
        and isinstance(source.get("satellite"), str)
        and _DIGITS.fullmatch(source["satellite"]) is not None,
        "receipt source record is invalid",
    )
    view = value.get("view")
    _require(
        isinstance(view, dict)
        and set(view) == {"H", "W", "fScale", "h", "sId", "sName", "w"}
        and isinstance(view.get("sId"), str)
        and _DIGITS.fullmatch(view["sId"]) is not None
        and type(view.get("W")) is int
        and type(view.get("H")) is int
        and type(view.get("w")) is int
        and type(view.get("h")) is int
        and view["W"] == assembled["width"]
        and view["H"] == assembled["height"]
        and view["w"] > 0
        and view["h"] > 0,
        "receipt view record is invalid",
    )
    grid = value.get("grid")
    expected_rows = math.ceil(view["H"] / view["h"])
    expected_columns = math.ceil(view["W"] / view["w"])
    _require(
        isinstance(grid, dict)
        and set(grid) == {"column_count", "coordinate_base", "row_count", "tile_count"}
        and grid.get("coordinate_base") == 1
        and grid.get("row_count") == expected_rows
        and grid.get("column_count") == expected_columns
        and grid.get("tile_count") == expected_rows * expected_columns,
        "receipt grid record is invalid",
    )
    tiles = value.get("tiles")
    _require(isinstance(tiles, list) and len(tiles) == grid["tile_count"], "receipt tiles are invalid")
    for index, tile in enumerate(tiles):
        row = index // expected_columns + 1
        column = index % expected_columns + 1
        _require(
            isinstance(tile, dict)
            and set(tile)
            == {
                "byte_count",
                "column",
                "content_type",
                "format",
                "height",
                "row",
                "sha256",
                "url",
                "width",
            }
            and tile.get("row") == row
            and tile.get("column") == column
            and tile.get("url") == _tile_url(project, view["sId"], source["satellite"], row, column)
            and tile.get("content_type") in _IMAGE_TYPES
            and tile.get("format") == _IMAGE_TYPES[tile["content_type"]]
            and type(tile.get("byte_count")) is int
            and tile["byte_count"] > 0
            and type(tile.get("width")) is int
            and type(tile.get("height")) is int
            and tile["width"] > 0
            and tile["height"] > 0
            and isinstance(tile.get("sha256"), str)
            and _SHA256.fullmatch(tile["sha256"]) is not None,
            f"receipt tile {row},{column} record is invalid",
        )
    recorded_path = assembled.get("path")
    _require(isinstance(recorded_path, str) and recorded_path, "receipt assembled image path is invalid")
    asset_path = Path(image_path) if image_path is not None else Path(recorded_path)
    try:
        payload = asset_path.read_bytes()
    except OSError as exc:
        raise PmcTileAssetError("assembled image is unavailable") from exc
    _require(_digest_bytes(payload) == assembled.get("sha256"), "assembled image digest mismatch")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            _require(image.format == "PNG", "assembled image format is invalid")
            _require(
                image.size == (assembled.get("width"), assembled.get("height")),
                "assembled image dimensions mismatch",
            )
            _require(image.mode == assembled.get("mode"), "assembled image mode mismatch")
            _require(_digest_bytes(image.tobytes()) == assembled.get("pixel_sha256"), "assembled pixel digest mismatch")
    except (UnidentifiedImageError, OSError) as exc:
        raise PmcTileAssetError("assembled image is not a valid PNG") from exc
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-receipt", type=Path)
    args = parser.parse_args()
    acquire(
        args.manifest_url,
        args.view_id,
        args.output,
        args.receipt,
        expected_receipt=args.expected_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
