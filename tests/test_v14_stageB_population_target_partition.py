from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research/specs/v14_snr_stageB_population_target_partition_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partition_is_prospective_value_blind_and_source_compatible() -> None:
    spec = json.loads(PATH.read_bytes())
    assert spec["status"] == "preregistered_before_blind_extraction_results_inspected"
    assert spec["scientific_verdict"] is None
    assert spec["optimization_allowed"] is False
    assignment = spec["assignment"]
    assert assignment["uses_x_value"] is False
    assert assignment["uses_y_value"] is False
    assert assignment["uses_uncertainty"] is False
    assert assignment["uses_model_output"] is False
    assert assignment["required_nonempty_partitions_per_panel"] == [
        "calibration",
        "validation",
        "held_out",
    ]
    assert spec["custody"]["combined_packet_allowed"] is False
    assert spec["custody"]["reuse_after_held_out_failure"] is False


def test_partition_authority_bindings_are_current() -> None:
    spec = json.loads(PATH.read_bytes())
    for key in ("measurement_protocol", "supersedes"):
        binding = spec[key]
        assert _sha(ROOT / binding["path"]) == binding["sha256"]
