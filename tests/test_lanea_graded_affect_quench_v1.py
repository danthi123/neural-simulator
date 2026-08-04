import copy
import json
from pathlib import Path

import pytest

from research.runners import _lanea_graded_affect_quench_v1 as lanea


def _passing_metrics():
    return {
        "persistence_each_sign": [0.8, 0.8],
        "persistence_mean": 0.8,
        "nmda_off_persistence_max": 0.02,
        "magnitude_pearson": 0.8,
        "magnitude_span": 0.04,
        "magnitude_polarity_accuracy": 1.0,
        "sign_crossing_pearson": 0.8,
        "sign_accuracy": 0.9,
        "sign_crossing_count": 3,
        "zero_band_fraction": 0.2,
        "largest_step_fraction": 0.3,
        "eviction_ratio": 0.3,
        "reignition_ratio": 0.8,
        "quench_lesion_residual": 1.0,
        "quench_lesion_gap": 0.7,
        "quench_fs_during_rate": 0.2,
        "quench_fs_read_rate": 0.005,
        "quench_drive_at_read": 0.0,
    }


def _candidate(weight, passing=True):
    metrics = _passing_metrics()
    checks = lanea.evaluate_metrics(metrics, _spec()["thresholds"])
    if not passing:
        checks["magnitude_tracks"] = False
    return {"recurrent_weight": float(weight), "pass": passing, "checks": checks, "metrics": metrics}


def _spec():
    return lanea.load_spec()[0]


def test_locked_seed_derivation_and_partitions_are_valid():
    spec, digest = lanea.load_spec()
    assert len(digest) == 64
    assert set(spec["seeds"]["diagnostic"]).isdisjoint(spec["seeds"]["formal"])
    assert len(spec["seeds"]["diagnostic"]) == 2
    assert len(spec["seeds"]["formal"]) == 6


def test_create_only_writer_refuses_reuse(tmp_path):
    destination = tmp_path / "result.json"
    digest = lanea.write_create_only(destination, {"b": 2, "a": 1})
    assert len(digest) == 64
    assert json.loads(destination.read_text()) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        lanea.write_create_only(destination, {"replacement": True})
    assert json.loads(destination.read_text()) == {"a": 1, "b": 2}


def test_every_locked_gate_is_load_bearing():
    spec = _spec()
    thresholds = spec["thresholds"]
    passing = _passing_metrics()
    assert all(lanea.evaluate_metrics(passing, thresholds).values())

    failure_values = {
        "persistence_each_sign": [0.49, 0.8],
        "persistence_mean": 0.61,
        "nmda_off_persistence_max": 0.11,
        "magnitude_pearson": 0.59,
        "magnitude_span": 0.019,
        "magnitude_polarity_accuracy": 0.87,
        "sign_crossing_pearson": 0.59,
        "sign_accuracy": 0.74,
        "sign_crossing_count": 1,
        "zero_band_fraction": 0.51,
        "largest_step_fraction": 0.61,
        "eviction_ratio": 0.61,
        "reignition_ratio": 0.59,
        "quench_lesion_residual": 0.89,
        "quench_lesion_gap": 0.29,
        "quench_fs_during_rate": 0.09,
        "quench_fs_read_rate": 0.021,
        "quench_drive_at_read": 0.000002,
    }
    baseline_checks = lanea.evaluate_metrics(passing, thresholds)
    for metric, value in failure_values.items():
        changed = copy.deepcopy(passing)
        changed[metric] = value
        checks = lanea.evaluate_metrics(changed, thresholds)
        failed = {name for name, ok in checks.items() if not ok}
        assert len(failed) == 1, (metric, failed, baseline_checks)


def test_diagnostic_selection_requires_both_seeds_and_breaks_ties_low():
    spec = _spec()
    rows = []
    for seed in spec["seeds"]["diagnostic"]:
        cells = []
        for weight in spec["diagnostic"]["recurrent_weight_ladder"]:
            cells.append(_candidate(weight, passing=weight in (16.0, 18.0)))
        rows.append({"seed": seed, "candidates": cells})
    result = lanea.select_diagnostic(rows, spec)
    assert result["selection_status"] == "SELECTED_FOR_FORMAL"
    assert result["selected_recurrent_weight"] == 16.0

    with pytest.raises(lanea.ProtocolError):
        lanea.select_diagnostic(rows[:1], spec)


def test_no_diagnostic_pass_keeps_formal_sealed():
    spec = _spec()
    rows = [{"seed": seed,
             "candidates": [_candidate(weight, passing=False)
                            for weight in spec["diagnostic"]["recurrent_weight_ladder"]]}
            for seed in spec["seeds"]["diagnostic"]]
    result = lanea.select_diagnostic(rows, spec)
    assert result["selected_recurrent_weight"] is None
    assert result["selection_status"] == "NO_SELECTION_FORMAL_REMAINS_SEALED"


def test_formal_verdict_requires_complete_partition_and_five_of_six():
    spec = _spec()
    rows = []
    for index, seed in enumerate(spec["seeds"]["formal"]):
        rows.append({"seed": seed, "selected_recurrent_weight": 16.0,
                     "candidate": _candidate(16.0, passing=index < 5)})
    assert lanea.formal_verdict(rows, spec)["verdict"] == "FORMAL_GO"
    rows[4]["candidate"] = _candidate(16.0, passing=False)
    assert lanea.formal_verdict(rows, spec)["verdict"] == "FORMAL_NO_GO"
    with pytest.raises(lanea.ProtocolError):
        lanea.formal_verdict(rows[:-1], spec)


def test_unassigned_seed_is_rejected_before_source_or_brain(monkeypatch):
    spec, digest = lanea.load_spec()
    monkeypatch.setattr(lanea, "_git_source_state", lambda: pytest.fail("source should not be inspected"))
    monkeypatch.setattr(lanea, "run_candidate", lambda *args: pytest.fail("brain should not run"))
    with pytest.raises(lanea.ProtocolError, match="not assigned"):
        lanea.execute_seed(spec, digest, "diagnostic", 123456)


def test_formal_recomputes_diagnostic_selection_before_opening(monkeypatch, tmp_path):
    spec, digest = lanea.load_spec()
    spec = copy.deepcopy(spec)
    spec["output_root"] = str(tmp_path.relative_to(lanea.REPO)) if tmp_path.is_relative_to(lanea.REPO) else str(tmp_path)
    rows = []
    for seed in spec["seeds"]["diagnostic"]:
        row = {"schema_version": 1, "phase": "diagnostic", "seed": seed, "spec_sha256": digest,
               "candidates": [_candidate(weight, passing=weight == 16.0)
                              for weight in spec["diagnostic"]["recurrent_weight_ladder"]]}
        lanea.write_create_only(lanea._artifact_path(spec, "diagnostic", seed), row)
        rows.append(row)
    selected = lanea.select_diagnostic(rows, spec)
    hashes = {str(row["seed"]): lanea._sha256_bytes(
        lanea._artifact_path(spec, "diagnostic", row["seed"]).read_bytes()) for row in rows}
    aggregate = {"phase": "diagnostic", "spec_sha256": digest, "input_artifact_sha256": hashes, **selected}
    aggregate["selected_recurrent_weight"] = 18.0
    lanea.write_create_only(lanea._artifact_path(spec, "diagnostic"), aggregate)
    with pytest.raises(lanea.ProtocolError, match="does not match sealed inputs"):
        lanea.validate_diagnostic_aggregate(spec, digest)


def test_spec_output_paths_are_new_and_phase_separated():
    spec = _spec()
    diagnostic = lanea._artifact_path(spec, "diagnostic", spec["seeds"]["diagnostic"][0])
    formal = lanea._artifact_path(spec, "formal", spec["seeds"]["formal"][0])
    assert diagnostic != formal
    assert "graded_quench_v1/diagnostic" in diagnostic.as_posix()
    assert "graded_quench_v1/formal" in formal.as_posix()
    assert not Path(diagnostic).exists()
    assert not Path(formal).exists()
