"""Deterministic, authenticated storage for one-dimensional simulation traces.

The format is a small ZIP archive containing a canonical JSON manifest and
three NumPy ``.npy`` members.  ZIP metadata, member order, JSON formatting,
and NumPy headers are fixed so equivalent traces produce identical bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile
import zlib

import numpy as np


__all__ = [
    "CompactTraceError",
    "SCHEMA_VERSION",
    "load_compact_trace",
    "load_trace",
    "read_compact_trace",
    "save_compact_trace",
    "save_trace",
    "write_compact_trace",
]


SCHEMA_VERSION = 1
_FORMAT = "neural-simulator.compact-trace"
_ARRAY_NAMES = ("time", "voltage", "spikes")
_MEMBER_NAMES = ("manifest.json", "time.npy", "voltage.npy", "spikes.npy")
_DTYPES = {
    "time": np.dtype("<f8"),
    "voltage": np.dtype("<f8"),
    "spikes": np.dtype("|b1"),
}


class CompactTraceError(ValueError):
    """Raised when a compact trace is invalid or fails authentication."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompactTraceError("manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CompactTraceError(f"manifest contains invalid JSON number: {value}")


def _as_trace_array(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:  # normalize array-protocol failures at the API boundary
        raise TypeError(f"{name} is not an array-like value") from exc

    expected = _DTYPES[name]
    if array.dtype != expected:
        raise TypeError(
            f"{name} must have dtype {expected.str}, got {array.dtype.str}"
        )
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if name != "spikes" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=expected)


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_for(member_bytes: Mapping[str, bytes], length: int) -> dict[str, Any]:
    arrays = []
    for name in _ARRAY_NAMES:
        payload = member_bytes[name]
        arrays.append(
            {
                "dtype": _DTYPES[name].str,
                "name": name,
                "nbytes": length * _DTYPES[name].itemsize,
                "sha256": _sha256(payload),
                "shape": [length],
            }
        )

    core = {
        "arrays": arrays,
        "format": _FORMAT,
        "schema": SCHEMA_VERSION,
    }
    return {**core, "binding": _sha256(_canonical_json(core))}


def _archive_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    member_bytes = {name: _npy_bytes(arrays[name]) for name in _ARRAY_NAMES}
    manifest = _manifest_for(member_bytes, len(arrays["time"]))

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        contents = {"manifest.json": _canonical_json(manifest), **{
            f"{name}.npy": member_bytes[name] for name in _ARRAY_NAMES
        }}
        for member_name in _MEMBER_NAMES:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.extra = b""
            info.comment = b""
            info.flag_bits = 0
            archive.writestr(info, contents[member_name], compresslevel=9)
    return output.getvalue()


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # A hard-link creation is atomic and fails instead of replacing an
        # existing destination, including when it is a symlink.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def save_compact_trace(
    path: str | os.PathLike[str],
    time: Any,
    voltage: Any,
    spikes: Any,
) -> str:
    """Write a trace once and return its SHA-256 digest.

    ``time`` and ``voltage`` must be finite little-endian ``float64`` vectors;
    ``spikes`` must be a boolean vector.  All three vectors must have the same
    length.  A destination that already exists is never overwritten.
    """
    arrays = {
        "time": _as_trace_array("time", time),
        "voltage": _as_trace_array("voltage", voltage),
        "spikes": _as_trace_array("spikes", spikes),
    }
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("time, voltage, and spikes must have the same length")

    payload = _archive_bytes(arrays)
    _publish_once(Path(path), payload)
    return _sha256(payload)


def _validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise CompactTraceError("manifest must be a JSON object")
    if set(manifest) != {"arrays", "binding", "format", "schema"}:
        raise CompactTraceError("manifest schema mismatch")
    if manifest["format"] != _FORMAT or manifest["schema"] != SCHEMA_VERSION:
        raise CompactTraceError("manifest schema mismatch")
    arrays = manifest["arrays"]
    if not isinstance(arrays, list) or len(arrays) != len(_ARRAY_NAMES):
        raise CompactTraceError("manifest must describe exactly three arrays")

    expected_keys = {"dtype", "name", "nbytes", "sha256", "shape"}
    names = []
    for descriptor in arrays:
        if not isinstance(descriptor, dict) or set(descriptor) != expected_keys:
            raise CompactTraceError("array descriptor schema mismatch")
        name = descriptor["name"]
        names.append(name)
        if name not in _ARRAY_NAMES or descriptor["dtype"] != _DTYPES[name].str:
            raise CompactTraceError("array descriptor type mismatch")
        shape = descriptor["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or shape[0] < 0
        ):
            raise CompactTraceError("array descriptor shape mismatch")
        if descriptor["nbytes"] != shape[0] * _DTYPES[name].itemsize:
            raise CompactTraceError("array descriptor byte-size mismatch")
        digest = descriptor["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != hashlib.sha256().digest_size * 2
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CompactTraceError("array descriptor digest mismatch")

    if tuple(names) != _ARRAY_NAMES:
        raise CompactTraceError("manifest array order or names mismatch")
    core = {key: manifest[key] for key in ("arrays", "format", "schema")}
    if not isinstance(manifest["binding"], str) or not hmac_compare(
        manifest["binding"], _sha256(_canonical_json(core))
    ):
        raise CompactTraceError("manifest binding mismatch")
    return arrays


def hmac_compare(left: str, right: str) -> bool:
    """Constant-time comparison for hexadecimal digest strings."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return hmac.compare_digest(left, right)
    except TypeError:
        return False


def load_compact_trace(
    path: str | os.PathLike[str],
    expected_sha256: str | None = None,
    *,
    expected_digest: str | None = None,
) -> dict[str, np.ndarray]:
    """Safely load and authenticate a compact trace.

    ``expected_sha256`` (or its alias ``expected_digest``) can bind the whole
    archive to a digest held outside the archive.  Without it, the manifest
    binding and every member digest still protect the archive's internal
    consistency and reject ordinary payload tampering.
    """
    if expected_sha256 is not None and expected_digest is not None:
        raise TypeError("provide only one expected archive digest")
    expected = expected_sha256 if expected_sha256 is not None else expected_digest
    archive_path = Path(path)
    try:
        payload = archive_path.read_bytes()
    except OSError:
        raise
    if expected is not None and not isinstance(expected, str):
        raise TypeError("expected archive digest must be a hexadecimal string")
    if expected is not None and not hmac_compare(_sha256(payload), expected.lower()):
        raise CompactTraceError("archive digest mismatch")

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise CompactTraceError("invalid compact trace archive") from exc

    with archive:
        try:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != list(_MEMBER_NAMES) or len(set(names)) != len(names):
                raise CompactTraceError("archive members are not the required unique set")
            if any(info.is_dir() or info.flag_bits & 0x1 for info in infos):
                raise CompactTraceError("archive contains an invalid member")
            if any(info.compress_type != zipfile.ZIP_DEFLATED for info in infos):
                raise CompactTraceError("archive compression mismatch")

            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(
                manifest_bytes.decode("ascii"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, UnicodeError) as exc:
            raise CompactTraceError("invalid manifest encoding") from exc
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
            raise CompactTraceError("invalid compact trace archive") from exc
        if _canonical_json(manifest) != manifest_bytes:
            raise CompactTraceError("manifest is not canonical")
        descriptors = _validate_manifest(manifest)

        result: dict[str, np.ndarray] = {}
        for descriptor in descriptors:
            name = descriptor["name"]
            member_name = f"{name}.npy"
            member_bytes = archive.read(member_name)
            if not hmac_compare(_sha256(member_bytes), descriptor["sha256"]):
                raise CompactTraceError(f"{name} member digest mismatch")
            try:
                array = np.load(io.BytesIO(member_bytes), allow_pickle=False)
            except (EOFError, OSError, ValueError, TypeError) as exc:
                raise CompactTraceError(f"invalid {name} NumPy member") from exc
            if not isinstance(array, np.ndarray):
                raise CompactTraceError(f"{name} member is not an ndarray")
            expected_dtype = _DTYPES[name]
            if (
                array.dtype != expected_dtype
                or array.shape != (descriptor["shape"][0],)
                or array.nbytes != descriptor["nbytes"]
            ):
                raise CompactTraceError(f"{name} dtype or shape mismatch")
            if name != "spikes" and not np.isfinite(array).all():
                raise CompactTraceError(f"{name} contains NaN or Inf")
            if _npy_bytes(np.ascontiguousarray(array, dtype=expected_dtype)) != member_bytes:
                raise CompactTraceError(f"{name} member is not canonical")
            result[name] = np.array(array, dtype=expected_dtype, copy=True)
        if _archive_bytes(result) != payload:
            raise CompactTraceError("archive is not canonical")
    return result


save_trace = save_compact_trace
write_compact_trace = save_compact_trace
load_trace = load_compact_trace
read_compact_trace = load_compact_trace
