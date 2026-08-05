import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from tools.diagnostic_trace import (
    DiagnosticTraceError,
    load_diagnostic_trace,
    save_diagnostic_trace,
)


@pytest.fixture()
def trace():
    return (
        np.array([0.05, 0.10, 0.15], dtype="<f8"),
        {
            "spikes": np.array([False, True, False], dtype="|b1"),
            "i_nap": np.array([-1.5, -2.0, 0.0], dtype="<f4"),
        },
    )


def test_round_trip_is_deterministic_and_manifest_is_canonical(tmp_path, trace):
    time, channels = trace
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_digest = save_diagnostic_trace(first, time, channels)
    second_digest = save_diagnostic_trace(second, time, reversed(list(channels.items())))

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest == hashlib.sha256(first.read_bytes()).hexdigest()
    loaded_time, loaded_channels = load_diagnostic_trace(first, first_digest)
    np.testing.assert_array_equal(loaded_time, time)
    assert list(loaded_channels) == ["i_nap", "spikes"]
    for name in channels:
        np.testing.assert_array_equal(loaded_channels[name], channels[name])

    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "manifest.json", "time.npy", "channels/i_nap.npy", "channels/spikes.npy"
        ]
        manifest_bytes = archive.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        assert manifest_bytes == json.dumps(
            manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        assert manifest["format"] == "neural-simulator.diagnostic-trace"
        assert manifest["samples"] == 3
        assert len(manifest["binding"]) == 64


@pytest.mark.parametrize(
    ("time", "channels", "error"),
    [
        (np.array([0], dtype="<f4"), {"x": np.zeros(1, "<f4")}, TypeError),
        (np.array([[0]], dtype="<f8"), {"x": np.zeros(1, "<f4")}, ValueError),
        (np.array([], dtype="<f8"), {"x": np.zeros(0, "<f4")}, ValueError),
        (np.array([1, 1], dtype="<f8"), {"x": np.zeros(2, "<f4")}, ValueError),
        (np.array([np.inf], dtype="<f8"), {"x": np.zeros(1, "<f4")}, ValueError),
        (np.array([0], dtype="<f8"), {}, ValueError),
        (np.array([0], dtype="<f8"), {"badName": np.zeros(1, "<f4")}, ValueError),
        (np.array([0], dtype="<f8"), {"time": np.zeros(1, "<f4")}, ValueError),
        (np.array([0], dtype="<f8"), {"x": np.zeros(1, "<f8")}, TypeError),
        (np.array([0], dtype="<f8"), {"x": np.zeros((1, 1), "<f4")}, ValueError),
        (np.array([0], dtype="<f8"), {"x": np.zeros(2, "<f4")}, ValueError),
        (np.array([0], dtype="<f8"), {"x": np.array([np.nan], "<f4")}, ValueError),
    ],
)
def test_rejects_invalid_inputs(tmp_path, time, channels, error):
    with pytest.raises(error):
        save_diagnostic_trace(tmp_path / "trace.zip", time, channels)


def test_rejects_duplicate_channel_and_existing_path(tmp_path, trace):
    time, channels = trace
    duplicate = [("signal", channels["i_nap"]), ("signal", channels["i_nap"])]
    with pytest.raises(ValueError, match="duplicate"):
        save_diagnostic_trace(tmp_path / "duplicate.zip", time, duplicate)
    path = tmp_path / "trace.zip"
    path.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        save_diagnostic_trace(path, time, channels)
    assert path.read_bytes() == b"keep me"


def test_load_rejects_wrong_digest_and_noncanonical_digest(tmp_path, trace):
    path = tmp_path / "trace.zip"
    digest = save_diagnostic_trace(path, *trace)
    with pytest.raises(DiagnosticTraceError, match="SHA-256 mismatch"):
        load_diagnostic_trace(path, "0" * 64)
    with pytest.raises(DiagnosticTraceError, match="lowercase"):
        load_diagnostic_trace(path, digest.upper())


def test_load_rejects_noncanonical_members_and_metadata(tmp_path, trace):
    original = tmp_path / "original.zip"
    save_diagnostic_trace(original, *trace)
    for mutation in ("member", "metadata"):
        target = tmp_path / f"{mutation}.zip"
        output = io.BytesIO()
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(output, "w") as archive:
            for info in source.infolist():
                clone = zipfile.ZipInfo(info.filename, info.date_time)
                clone.compress_type = info.compress_type
                clone.create_system = info.create_system
                clone.external_attr = info.external_attr
                if mutation == "metadata" and info.filename == "time.npy":
                    clone.date_time = (1981, 1, 1, 0, 0, 0)
                archive.writestr(clone, source.read(info.filename))
            if mutation == "member":
                archive.writestr("extra.txt", b"unexpected")
        target.write_bytes(output.getvalue())
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        with pytest.raises(DiagnosticTraceError, match="noncanonical"):
            load_diagnostic_trace(target, digest)
