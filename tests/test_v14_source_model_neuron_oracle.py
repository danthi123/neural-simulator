import hashlib
import json
from pathlib import Path

import pytest

from research.runners import v14_source_model_neuron_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/specs/v14_khaliq_native_initialization_oracle_v1.json"


def test_repository_oracle_contract_authenticates_source_runner_and_commands():
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    document, binding = oracle._load_spec(SPEC, digest, ROOT)

    assert binding == {
        "path": "research/specs/v14_khaliq_native_initialization_oracle_v1.json",
        "sha256": digest,
    }
    assert document["source"]["sha256"] == (
        "1a3382714bd0962665ec31f7dfac2aa3a9e403a5e3d23e29851afec232c4543e"
    )
    runner = ROOT / document["runner"]["path"]
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == document["runner"]["sha256"]
    assert document["decision_rules"]["SOURCE_NATIVE_INITIALIZATION_DEFECT"]


def test_wrong_contract_digest_fails_before_source_download():
    with pytest.raises(oracle.OracleError, match="digest mismatch"):
        oracle._load_spec(SPEC, "0" * 64, ROOT)


def test_semantic_digest_excludes_only_self_digest():
    document = {"schema": oracle.OUTPUT_SCHEMA, "value": 3, "sha256": "old"}
    expected = hashlib.sha256(
        json.dumps(
            {"schema": oracle.OUTPUT_SCHEMA, "value": 3},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert oracle._semantic_digest(document) == expected
