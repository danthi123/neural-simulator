"""Deterministic authenticated archives for multichannel diagnostic traces."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np


ChannelItems: TypeAlias = Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]]

SCHEMA_VERSION = 1
_FORMAT = "neural-simulator.diagnostic-trace"
_NAME_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", re.ASCII)
_RESERVED_NAMES = frozenset({"manifest", "time"})
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_FILE_MODE = 0o100600 << 16
_MANIFEST_NAME = "manifest.json"
_TIME_NAME = "time.npy"


class DiagnosticTraceError(ValueError):
    """Raised when a diagnostic trace is invalid or fails authentication."""


def _array_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = _FILE_MODE
    info.extra = b""
    info.comment = b""
    info.flag_bits = 0
    return info


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_channels(channels: ChannelItems) -> list[tuple[str, np.ndarray]]:
    if isinstance(channels, Mapping):
        items = list(channels.items())
    else:
        try:
            items = list(channels)
        except TypeError as exc:
            raise TypeError("channels must be a mapping or iterable of pairs") from exc
    if not items:
        raise ValueError("at least one diagnostic channel is required")

    result: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each channel must be a (name, array) pair")
        name, array = item
        if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid channel name: {name!r}")
        if name in _RESERVED_NAMES:
            raise ValueError(f"reserved channel name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate channel name: {name!r}")
        seen.add(name)
        result.append((name, array))
    return sorted(result, key=lambda pair: pair[0])


def _validate_arrays(
    time: np.ndarray, channels: ChannelItems
) -> list[tuple[str, np.ndarray]]:
    if not isinstance(time, np.ndarray) or time.dtype != np.dtype("<f8"):
        raise TypeError("time must be a little-endian float64 numpy array")
    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a nonempty one-dimensional array")
    if not np.isfinite(time).all() or (time.size > 1 and not np.all(np.diff(time) > 0.0)):
        raise ValueError("time must be finite and strictly increasing")

    items = _normalise_channels(channels)
    for name, array in items:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"channel {name!r} must be a numpy array")
        if array.dtype not in (np.dtype("<f4"), np.dtype("|b1")):
            raise TypeError(
                f"channel {name!r} must have little-endian float32 or bool dtype"
            )
        if array.ndim != 1:
            raise ValueError(f"channel {name!r} must be one-dimensional")
        if len(array) != len(time):
            raise ValueError(f"channel {name!r} length does not match time")
        if array.dtype == np.dtype("<f4") and not np.isfinite(array).all():
            raise ValueError(f"channel {name!r} contains nonfinite values")
    return items


def _archive_bytes(time: np.ndarray, channels: ChannelItems) -> bytes:
    items = _validate_arrays(time, channels)
    time_data = _array_bytes(np.ascontiguousarray(time, dtype="<f8"))
    payloads: list[tuple[str, bytes]] = []
    channel_manifest: list[dict[str, object]] = []

    for name, array in items:
        member = f"channels/{name}.npy"
        data = _array_bytes(np.ascontiguousarray(array))
        payloads.append((member, data))
        channel_manifest.append(
            {
                "dtype": array.dtype.str,
                "name": name,
                "path": member,
                "sha256": _sha256(data),
                "shape": [len(array)],
            }
        )

    core = {
        "channels": channel_manifest,
        "format": _FORMAT,
        "samples": len(time),
        "schema": SCHEMA_VERSION,
        "time": {
            "dtype": time.dtype.str,
            "path": _TIME_NAME,
            "sha256": _sha256(time_data),
            "shape": [len(time)],
        },
    }
    manifest = {**core, "binding": _sha256(_canonical_json(core))}

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.comment = b""
        members = [(_MANIFEST_NAME, _canonical_json(manifest)), (_TIME_NAME, time_data)]
        members.extend(payloads)
        for name, data in members:
            archive.writestr(_zip_info(name), data, compresslevel=9)
    return output.getvalue()


def _publish_once(path: Path, data: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_diagnostic_trace(
    path: str | os.PathLike[str], time: np.ndarray, channels: ChannelItems
) -> str:
    """Publish one canonical archive without replacing an existing path."""
    data = _archive_bytes(time, channels)
    _publish_once(Path(path), data)
    return _sha256(data)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticTraceError(f"duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _read_array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    try:
        data = archive.read(name)
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (KeyError, ValueError, EOFError) as exc:
        raise DiagnosticTraceError(f"invalid array member: {name!r}") from exc
    if not isinstance(array, np.ndarray):
        raise DiagnosticTraceError(f"invalid array member: {name!r}")
    return array


def _parse_archive(data: bytes) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise DiagnosticTraceError("duplicate ZIP member")
            if archive.comment:
                raise DiagnosticTraceError("noncanonical ZIP comment")
            for info in infos:
                if (
                    info.date_time != _ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.create_system != 3
                    or info.external_attr != _FILE_MODE
                    or info.extra
                    or info.comment
                ):
                    raise DiagnosticTraceError(
                        f"noncanonical ZIP metadata: {info.filename!r}"
                    )
            try:
                raw_manifest = archive.read(_MANIFEST_NAME)
                manifest = json.loads(
                    raw_manifest.decode("ascii"), object_pairs_hook=_unique_object
                )
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DiagnosticTraceError("invalid manifest") from exc
            if not isinstance(manifest, dict):
                raise DiagnosticTraceError("invalid manifest")
            time = _read_array(archive, _TIME_NAME)
            raw_channels = manifest.get("channels")
            if not isinstance(raw_channels, list):
                raise DiagnosticTraceError("invalid manifest channels")
            channels: list[tuple[str, np.ndarray]] = []
            for entry in raw_channels:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise DiagnosticTraceError("invalid channel manifest entry")
                name = entry["name"]
                channels.append((name, _read_array(archive, f"channels/{name}.npy")))
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise DiagnosticTraceError("invalid diagnostic trace archive") from exc

    canonical = _archive_bytes(time, channels)
    if not hmac.compare_digest(data, canonical):
        raise DiagnosticTraceError("noncanonical or tampered diagnostic trace archive")
    return time, dict(channels)


def load_diagnostic_trace(
    path: str | os.PathLike[str], expected_sha256: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Authenticate and load one canonical diagnostic trace archive."""
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise DiagnosticTraceError(
            "expected_sha256 must be a lowercase SHA-256 digest"
        )
    data = Path(path).read_bytes()
    if not hmac.compare_digest(_sha256(data), expected_sha256):
        raise DiagnosticTraceError("diagnostic trace SHA-256 mismatch")
    return _parse_archive(data)


__all__ = [
    "DiagnosticTraceError",
    "SCHEMA_VERSION",
    "load_diagnostic_trace",
    "save_diagnostic_trace",
]
