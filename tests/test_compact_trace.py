import io
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from tools.compact_trace import (
    CompactTraceError,
    load_compact_trace,
    save_compact_trace,
)


TIME = np.array([0.0, 0.125, 1.5, 2.0], dtype=np.float64)
VOLTAGE = np.array([-65.25, -64.0, 3.125, 8.5], dtype=np.float64)
SPIKES = np.array([False, True, False, True], dtype=np.bool_)


def _save(path: Path) -> str:
    return save_compact_trace(path, TIME, VOLTAGE, SPIKES)


def _members(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _write_members(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload, compresslevel=1)


def test_round_trip_preserves_exact_values_and_dtypes(tmp_path):
    path = tmp_path / "trace.ct"
    digest = _save(path)

    loaded = load_compact_trace(path, expected_sha256=digest)

    assert set(loaded) == {"time", "voltage", "spikes"}
    assert loaded["time"].dtype == np.dtype("<f8")
    assert loaded["voltage"].dtype == np.dtype("<f8")
    assert loaded["spikes"].dtype == np.dtype("|b1")
    assert np.array_equal(loaded["time"], TIME)
    assert np.array_equal(loaded["voltage"], VOLTAGE)
    assert np.array_equal(loaded["spikes"], SPIKES)


def test_serialization_is_byte_deterministic(tmp_path):
    first = tmp_path / "first.ct"
    second = tmp_path / "second.ct"

    first_digest = _save(first)
    second_digest = _save(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest


def test_loading_does_not_depend_on_local_deflate_reserialization(tmp_path, monkeypatch):
    path = tmp_path / "trace.ct"
    digest = _save(path)

    monkeypatch.setattr(
        "tools.compact_trace._archive_bytes",
        lambda arrays: (_ for _ in ()).throw(AssertionError("must not recompress on load")),
    )

    loaded = load_compact_trace(path, expected_sha256=digest)
    assert np.array_equal(loaded["voltage"], VOLTAGE)


def test_rejects_noncanonical_zip_metadata(tmp_path):
    source = tmp_path / "source.ct"
    broken = tmp_path / "broken.ct"
    _save(source)
    with zipfile.ZipFile(broken, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in _members(source):
            archive.writestr(name, payload)

    with pytest.raises(CompactTraceError, match="metadata is not canonical"):
        load_compact_trace(broken)


def test_write_once_rejects_existing_destination(tmp_path):
    path = tmp_path / "trace.ct"
    _save(path)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        _save(path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_external_digest_rejects_archive_tampering(tmp_path):
    path = tmp_path / "trace.ct"
    digest = _save(path)
    path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))

    with pytest.raises(CompactTraceError, match="archive digest mismatch"):
        load_compact_trace(path, expected_digest=digest)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda names: names + [("extra.npy", b"junk")], "required unique set"),
        (lambda names: names[:1] + [names[1], names[1]] + names[2:], "required unique set"),
        (lambda names: [("../time.npy", payload) if name == "time.npy" else (name, payload) for name, payload in names], "required unique set"),
    ],
)
def test_rejects_extra_duplicate_and_path_traversal_members(tmp_path, mutation, message):
    source = tmp_path / "source.ct"
    broken = tmp_path / "broken.ct"
    _save(source)
    _write_members(broken, mutation(_members(source)))

    with pytest.raises(CompactTraceError, match=message):
        load_compact_trace(broken)


def test_rejects_wrong_dtype_and_shape_even_with_rebound_manifest(tmp_path):
    source = tmp_path / "source.ct"
    wrong_dtype = tmp_path / "wrong-dtype.ct"
    wrong_shape = tmp_path / "wrong-shape.ct"
    _save(source)
    members = dict(_members(source))

    wrong_voltage = io.BytesIO()
    np.save(wrong_voltage, VOLTAGE.astype(np.float32), allow_pickle=False)
    members["voltage.npy"] = wrong_voltage.getvalue()
    _write_with_rebound_manifest(wrong_dtype, members)

    wrong_time = io.BytesIO()
    np.save(wrong_time, TIME.reshape(2, 2), allow_pickle=False)
    members["time.npy"] = wrong_time.getvalue()
    _write_with_rebound_manifest(wrong_shape, members)

    with pytest.raises(CompactTraceError, match="type mismatch|dtype or shape mismatch"):
        load_compact_trace(wrong_dtype)
    with pytest.raises(CompactTraceError, match="type mismatch|dtype or shape mismatch"):
        load_compact_trace(wrong_shape)


def _write_with_rebound_manifest(path: Path, members: dict[str, bytes]) -> None:
    manifest = json.loads(members["manifest.json"])
    for descriptor in manifest["arrays"]:
        member = f"{descriptor['name']}.npy"
        descriptor["sha256"] = hashlib.sha256(members[member]).hexdigest()
    core = {key: manifest[key] for key in ("arrays", "format", "schema")}
    manifest["binding"] = hashlib.sha256(
        json.dumps(core, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    members["manifest.json"] = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    _write_members(path, list(members.items()))


@pytest.mark.parametrize("field", ["schema", "format"])
def test_rejects_schema_mismatch(tmp_path, field):
    source = tmp_path / "source.ct"
    broken = tmp_path / f"broken-{field}.ct"
    _save(source)
    members = dict(_members(source))
    manifest = json.loads(members["manifest.json"])
    manifest[field] = 999 if field == "schema" else "other.format"
    members["manifest.json"] = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    _write_members(broken, list(members.items()))

    with pytest.raises(CompactTraceError, match="binding mismatch|schema mismatch"):
        load_compact_trace(broken)


@pytest.mark.parametrize("name", ["time", "voltage"])
def test_rejects_non_finite_values_on_save(name, tmp_path):
    values = {"time": TIME.copy(), "voltage": VOLTAGE.copy(), "spikes": SPIKES}
    values[name][1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        save_compact_trace(tmp_path / "bad.ct", **values)


def test_rejects_non_finite_values_on_load(tmp_path):
    source = tmp_path / "source.ct"
    broken = tmp_path / "broken.ct"
    _save(source)
    members = dict(_members(source))
    non_finite = io.BytesIO()
    bad_voltage = VOLTAGE.copy()
    bad_voltage[1] = np.inf
    np.save(non_finite, bad_voltage, allow_pickle=False)
    members["voltage.npy"] = non_finite.getvalue()
    _write_with_rebound_manifest(broken, members)

    with pytest.raises(CompactTraceError, match="NaN or Inf"):
        load_compact_trace(broken)


def test_rejects_pickle_payload(tmp_path):
    path = tmp_path / "pickle.ct"
    members = [
        ("manifest.json", b"{}"),
        ("time.npy", b"not-a-npy"),
        ("voltage.npy", b"not-a-npy"),
        ("spikes.npy", b"not-a-npy"),
    ]
    _write_members(path, members)

    with pytest.raises(CompactTraceError):
        load_compact_trace(path)
