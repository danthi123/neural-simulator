"""Tests for the exact V14 Stage B Sobol candidate generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.v14_stageB_candidate_batch import (
    EXACT_SCREEN_COUNT,
    MANIFEST_SCHEMA,
    StageBCandidateBatchError,
    build_candidate_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "research/specs/v14_snr_stageB_packet_template.json"
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


def test_filed_template_generates_exact_deterministic_seed_free_batch() -> None:
    first = build_candidate_manifest(TEMPLATE, TEMPLATE_SHA256, root=ROOT)
    second = build_candidate_manifest(TEMPLATE, TEMPLATE_SHA256, root=ROOT)

    assert first == second
    assert first["schema"] == MANIFEST_SCHEMA
    assert first["design"]["exact_count"] == EXACT_SCREEN_COUNT
    assert first["design"]["scientific_seed"] is None
    assert len(first["candidates"]) == EXACT_SCREEN_COUNT
    assert len({item["candidate_sha256"] for item in first["candidates"]}) == EXACT_SCREEN_COUNT
    assert [item["point_index"] for item in first["candidates"]] == list(range(EXACT_SCREEN_COUNT))
    assert all("seed" not in item["candidate"] for item in first["candidates"])


def test_every_generated_parameter_stays_inside_filed_transform_bound() -> None:
    manifest = build_candidate_manifest(TEMPLATE, TEMPLATE_SHA256, root=ROOT)
    space = {item["candidate_key"]: item for item in manifest["search_space"]}

    for row in manifest["candidates"]:
        parameters = row["candidate"]["parameters"]
        assert set(parameters) == set(space)
        for key, value in parameters.items():
            assert space[key]["low"] <= value <= space[key]["high"]

    first = manifest["candidates"][0]["candidate"]["parameters"]
    assert all(first[key] == specification["low"] for key, specification in space.items())


def test_digest_mismatch_and_noncanonical_template_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(StageBCandidateBatchError, match="digest does not match"):
        build_candidate_manifest(TEMPLATE, "0" * 64, root=ROOT)

    value = json.loads(TEMPLATE.read_text(encoding="ascii"))
    copied = tmp_path / "template.json"
    copied.write_text(json.dumps(value, indent=2), encoding="ascii")
    digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    with pytest.raises(StageBCandidateBatchError, match="inside the repository"):
        build_candidate_manifest(copied, digest, root=ROOT)


def test_manifest_writer_is_repository_scoped_and_write_once(tmp_path: Path) -> None:
    manifest = build_candidate_manifest(TEMPLATE, TEMPLATE_SHA256, root=ROOT)
    destination = ROOT / "research/findings/raw" / f"candidate-manifest-test-{tmp_path.name}.json"
    try:
        assert write_manifest(manifest, destination, root=ROOT) == destination
        stored = json.loads(destination.read_text(encoding="ascii"))
        assert stored == manifest
        with pytest.raises(StageBCandidateBatchError, match="refusing to replace"):
            write_manifest(manifest, destination, root=ROOT)
        with pytest.raises(StageBCandidateBatchError, match="inside the repository"):
            write_manifest(manifest, tmp_path / "outside.json", root=ROOT)
    finally:
        destination.unlink(missing_ok=True)
