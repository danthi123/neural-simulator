from __future__ import annotations

import json
from pathlib import Path

from tools import v13_stage0_performance_confirmation_v8 as confirmation


def test_v8_readiness_rebinds_sealed_inputs() -> None:
    result = confirmation.readiness()
    assert result["status"] == "READY"
    assert result["candidate_revision"] == confirmation.CANDIDATE_REVISION
    assert result["stage1_seed_1031"] == "sealed-not-read-or-executed"


def test_v8_candidate_outputs_are_inside_the_raw_evidence_tree(tmp_path: Path) -> None:
    paths = confirmation._candidate_paths(tmp_path)
    assert paths["artifact"].parent == (
        tmp_path / confirmation.RAW_PATH / "candidate-runtime"
    )
    assert paths["receipt"].is_relative_to(tmp_path / confirmation.RAW_PATH)
    assert paths["sidecar"].is_relative_to(tmp_path / confirmation.RAW_PATH)


def test_v8_spec_digest_and_final_preconditions_are_structured() -> None:
    spec = json.loads((confirmation.ROOT / confirmation.SPEC_PATH).read_text())
    body = dict(spec)
    digest = body.pop("sha256")
    assert digest == confirmation.v7._canonical_digest(body)
    final = json.loads(
        (confirmation.ROOT / confirmation.RAW_PATH / "final-stage0-v8.json").read_text()
    )
    assert final["candidate_receipt_complete"] is True
    assert all(item["ok"] is True for item in final["preconditions"])
