"""Tests for exact-byte, fail-closed JSON evidence reads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import stable_json_evidence as stable_json


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_returns_value_exact_bytes_and_both_digest_domains(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    raw = '{\n  "z": "café",\n  "a": [2, 1]\n}\n'.encode()
    path.write_bytes(raw)

    evidence = stable_json.read_stable_json_evidence(path, require_object=True)

    assert evidence.path == path
    assert evidence.value == {"z": "café", "a": [2, 1]}
    assert evidence.raw_bytes == raw
    assert evidence.file_sha256 == hashlib.sha256(raw).hexdigest()
    assert evidence.canonical_json_sha256 == _canonical_digest(evidence.value)
    assert evidence.canonicalization == stable_json.CANONICALIZATION


def test_formatting_changes_only_the_exact_byte_digest(tmp_path: Path) -> None:
    compact = tmp_path / "compact.json"
    expanded = tmp_path / "expanded.json"
    compact.write_text('{"a":1,"b":2}', encoding="utf-8")
    expanded.write_text('{\n  "b": 2,\n  "a": 1\n}\n', encoding="utf-8")

    compact_evidence = stable_json.read_stable_json_evidence(compact)
    expanded_evidence = stable_json.read_stable_json_evidence(expanded)

    assert compact_evidence.file_sha256 != expanded_evidence.file_sha256
    assert (
        compact_evidence.canonical_json_sha256
        == expanded_evidence.canonical_json_sha256
    )


def test_non_object_is_allowed_unless_object_is_required(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text('[1, 2, 3]', encoding="utf-8")

    assert stable_json.read_stable_json_evidence(path).value == [1, 2, 3]
    with pytest.raises(stable_json.StableJsonEvidenceError, match="must be an object"):
        stable_json.read_stable_json_evidence(path, require_object=True)


@pytest.mark.parametrize("raw", [b'{"broken":', b"\xff", b'{"value": NaN}'])
def test_invalid_json_fails(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)

    with pytest.raises(stable_json.StableJsonEvidenceError, match="is invalid"):
        stable_json.read_stable_json_evidence(path)


def test_symlink_fails(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(
        stable_json.StableJsonEvidenceError,
        match="without following symlinks",
    ):
        stable_json.read_stable_json_evidence(link)


def test_non_regular_file_fails(tmp_path: Path) -> None:
    with pytest.raises(stable_json.StableJsonEvidenceError, match="not a regular file"):
        stable_json.read_stable_json_evidence(tmp_path)


def test_replacement_during_parse_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    replacement.write_text('{"version": 2}', encoding="utf-8")
    original_parse = stable_json._parse_json

    def replace_path(raw_bytes: bytes) -> object:
        value = original_parse(raw_bytes)
        os.replace(replacement, path)
        return value

    monkeypatch.setattr(stable_json, "_parse_json", replace_path)

    with pytest.raises(stable_json.StableJsonEvidenceError, match="changed"):
        stable_json.read_stable_json_evidence(path)


def test_in_place_mutation_during_parse_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    original_parse = stable_json._parse_json

    def mutate_file(raw_bytes: bytes) -> object:
        value = original_parse(raw_bytes)
        path.write_text('{"version": 2}', encoding="utf-8")
        return value

    monkeypatch.setattr(stable_json, "_parse_json", mutate_file)

    with pytest.raises(stable_json.StableJsonEvidenceError, match="changed while being parsed"):
        stable_json.read_stable_json_evidence(path)


def test_symlink_substitution_during_parse_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.json"
    target = tmp_path / "target.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    target.write_text('{"version": 1}', encoding="utf-8")
    original_parse = stable_json._parse_json

    def substitute_symlink(raw_bytes: bytes) -> object:
        value = original_parse(raw_bytes)
        path.unlink()
        path.symlink_to(target)
        return value

    monkeypatch.setattr(stable_json, "_parse_json", substitute_symlink)

    with pytest.raises(stable_json.StableJsonEvidenceError, match="changed|pathname"):
        stable_json.read_stable_json_evidence(path)


def test_disappearance_during_parse_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{}", encoding="utf-8")
    original_parse = stable_json._parse_json

    def remove_path(raw_bytes: bytes) -> object:
        value = original_parse(raw_bytes)
        path.unlink()
        return value

    monkeypatch.setattr(stable_json, "_parse_json", remove_path)

    with pytest.raises(stable_json.StableJsonEvidenceError, match="changed|disappeared"):
        stable_json.read_stable_json_evidence(path)
