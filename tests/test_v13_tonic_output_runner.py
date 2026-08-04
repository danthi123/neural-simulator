"""Runner-level checks for the preregistered V13 tonic-output protocol."""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from research.runners import _vocal_action_credit_gate_v13_tonic_output as v13  # noqa: E402
from sim.backend import to_host  # noqa: E402


def test_locked_spec_matches_runner_and_seed_partitions_do_not_overlap():
    spec = v13.load_locked_spec()
    assert spec["partitions"] == v13.PARTITIONS
    assigned = [seed for seeds in v13.PARTITIONS.values() for seed in seeds]
    assert len(assigned) == len(set(assigned))
    assert v13.PARTITIONS["reserved_for_stage1"] == [1031]


def test_earned_compatibility_correction_is_bound_to_committed_evidence():
    result = v13._load_compatibility_correction(v13.COMPATIBILITY_CORRECTION_PATH)
    assert result["outcome"] == "DETERMINISTIC_COMPATIBILITY_GO"
    assert result["deterministic_patch_id"] == v13.DETERMINISTIC_PATCH_ID
    assert result["baseline_bundle_present_in_candidate_source"] is True
    assert result["twin_intrinsic_states_valid"] is True


def test_compatibility_correction_refuses_noncanonical_artifact(tmp_path):
    substitute = tmp_path / "comparison.json"
    substitute.write_text(v13.COMPATIBILITY_CORRECTION_PATH.read_text())
    with pytest.raises(ValueError, match="must be"):
        v13._load_compatibility_correction(substitute)


def test_tonic_builder_has_only_immutable_intrinsic_drive():
    bridge = v13.build_tonic_bridge(7, 100.0)
    assert bridge.cp_intrinsic_current_pA is not None
    np.testing.assert_array_equal(
        to_host(bridge.cp_external_input_current), np.zeros(40, np.float32)
    )
    np.testing.assert_array_equal(
        to_host(bridge.cp_connections.data),
        np.zeros(bridge.cp_connections.nnz, np.float32),
    )
    audit = v13._population_audit(bridge)
    assert audit["pass"], audit


def test_inhibitory_builder_preserves_only_registered_functional_pathway():
    bridge = v13.build_inhibitory_bridge(7, 100.0)
    weights = np.asarray(to_host(bridge.cp_connections.data))
    assert set(np.unique(weights)) == {0.0, 8.0}
    assert np.count_nonzero(weights == 8.0) == 20 * 40
    assert bridge.cp_intrinsic_current_pA is not None
    np.testing.assert_array_equal(
        to_host(bridge.cp_intrinsic_current_pA)[:20], np.zeros(20, np.float32)
    )
    np.testing.assert_array_equal(
        to_host(bridge.cp_intrinsic_current_pA)[20:], np.full(40, 100.0, np.float32)
    )


def test_short_run_keeps_external_zero_and_intrinsic_immutable():
    bridge = v13.build_tonic_bridge(7, 100.0)
    run = v13._run_steps(bridge, 100)
    assert run["external_zero"]
    assert run["intrinsic_hash_before"] == run["intrinsic_hash_after"]
    assert run["weight_hash_before"] == run["weight_hash_after"]
    assert run["raster"].sum() > 0


def test_checkpoint_continuation_is_exact_on_test_seed():
    result = v13.run_checkpoint_gate(7, 100.0)
    assert result["pass"], result["checks"]


def test_merge_calibration_selects_lowest_common_passing_point(tmp_path):
    identity = {"runner": "same"}
    correction = v13._load_compatibility_correction(
        v13.COMPATIBILITY_CORRECTION_PATH
    )
    base = {
        "stage": "calibration_backend",
        "seed": 1013,
        "source_sha": "abc",
        "source_identity": identity,
        "compatibility_correction": correction,
        "rows": [{"current_pA": value} for value in v13.LADDER_PA],
    }
    numpy = {**base, "backend": "numpy", "passing_currents_pA": [100, 125]}
    cupy = {**base, "backend": "cupy", "passing_currents_pA": [100, 150]}
    numpy_path = tmp_path / "numpy.json"
    cupy_path = tmp_path / "cupy.json"
    numpy_path.write_text(json.dumps(numpy))
    cupy_path.write_text(json.dumps(cupy))
    merged = v13.merge_calibration(numpy_path, cupy_path)
    assert merged["calibration_go"]
    assert merged["selected_current_pA"] == 100


def test_merge_calibration_refuses_source_mismatch(tmp_path):
    rows = [{"current_pA": value} for value in v13.LADDER_PA]
    paths = []
    for backend, identity in (("numpy", {"x": "1"}), ("cupy", {"x": "2"})):
        path = tmp_path / f"{backend}.json"
        path.write_text(json.dumps({
            "backend": backend, "seed": 1013, "source_sha": "abc",
            "source_identity": identity, "rows": rows,
            "passing_currents_pA": [100],
        }))
        paths.append(path)
    with pytest.raises(ValueError, match="identical sealed sources"):
        v13.merge_calibration(*paths)


def test_merge_calibration_refuses_missing_compatibility_binding(tmp_path):
    identity = {"runner": "same"}
    paths = []
    for backend in ("numpy", "cupy"):
        path = tmp_path / f"{backend}.json"
        path.write_text(json.dumps({
            "backend": backend, "seed": 1013, "source_sha": "abc",
            "source_identity": identity,
            "rows": [{"current_pA": value} for value in v13.LADDER_PA],
            "passing_currents_pA": [100],
        }))
        paths.append(path)
    with pytest.raises(ValueError, match="compatibility correction"):
        v13.merge_calibration(*paths)


def test_final_merge_requires_every_backend_and_gate(tmp_path):
    artifacts = [v13.COMPATIBILITY_CORRECTION_PATH]
    rows = (
        ("replication", "numpy", True),
        ("replication", "cupy", True),
        ("held_out", "cupy", True),
        ("held_out", "numpy", True),
        ("performance", "cupy", True),
    )
    for index, (stage, backend, go) in enumerate(rows):
        path = tmp_path / f"{index}.json"
        payload = {
            "stage": stage,
            "backend": backend,
            "go": go,
            "outcome": "GO",
            "verdict_status": "GO" if go else "NO-GO",
            "preconditions": [{"name": "fixture", "ok": True}],
            "undefined_reasons": [],
        }
        if stage in ("replication", "held_out"):
            payload["selected_current_pA"] = 100
        path.write_text(json.dumps(payload))
        artifacts.append(path)
    merged = v13.merge_final(artifacts)
    assert merged["outcome"] == "TONIC_OUTPUT_GO"

    failed = json.loads(artifacts[-1].read_text())
    failed["go"] = False
    artifacts[-1].write_text(json.dumps(failed))
    merged = v13.merge_final(artifacts)
    assert merged["outcome"] == "TONIC_OUTPUT_NO_GO"


def test_final_merge_refuses_unearned_input_verdict(tmp_path):
    artifact = tmp_path / "invalid.json"
    artifact.write_text(json.dumps({"stage": "compatibility", "go": True}))
    with pytest.raises(ValueError, match="earned verdict"):
        v13.merge_final([artifact] * 6)


def test_compatibility_artifact_carries_earned_preconditions():
    result = v13.run_compatibility()
    assert result["preconditions"]
    assert all(item["ok"] is True for item in result["preconditions"])
    assert result["verdict_status"] in {"GO", "NO-GO"}
    assert result["intrinsic_is_none"] is result["checks"]["intrinsic_is_none"]
