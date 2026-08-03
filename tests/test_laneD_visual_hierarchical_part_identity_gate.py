"""Focused guards for the hierarchical V2-part to IT-identity successor."""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np
import pytest

from research.runners import _laneD_visual_hierarchical_part_identity_gate as gate


def _tiny_args(tmp_path):
    return gate.build_parser().parse_args(
        [
            "--phase", "smoke", "--seeds", str(gate.SMOKE_SEED),
            "--epochs", "1", "--train-frames", "3", "--tracks-per-object", "1",
            "--held-frames", "2", "--image-size", "16", "--n-pos", "4",
            "--gabor-rf-radius", "2", "--v1-active", "8", "--v1-fs-cells", "4",
            "--v2-grid", "2", "--rf-width", "3", "--v2-part-cells", "8",
            "--v2-fs-cells", "2", "--it-cells", "16", "--it-expected-active", "2",
            "--it-fs-cells", "4", "--latency-steps", "40",
            "--out", str(tmp_path / "smoke.json"),
        ]
    )


def test_seed_policy_requires_exact_aggregate_and_locks_later_phases():
    assert gate.SMOKE_SEED == 222
    assert gate.CALIBRATION_SEEDS == (503, 509)
    assert gate.DEVELOPMENT_SEEDS == (521, 523, 541)
    assert gate.HELDOUT_SEEDS == (547, 557, 563)
    gate.validate_seed_partition("smoke", [222])
    gate.validate_seed_partition("calibration", [503, 509])
    assert gate.validate_individual_seed("calibration", 503) == 503
    for invalid in ([503], [509, 503], [503, 503], [503, 509, 509]):
        with pytest.raises(ValueError, match="exact ordered seeds"):
            gate.validate_seed_partition("calibration", invalid)
    with pytest.raises(ValueError, match="locked"):
        gate.validate_seed_partition("development", [521, 523, 541])
    with pytest.raises(ValueError, match="locked"):
        gate.validate_individual_seed("heldout", 547)


def test_aggregate_verdict_requires_both_calibration_rows():
    assert gate.aggregate_formal_verdict("smoke", []) == "NOT-SCIENTIFIC-SMOKE"
    assert gate.aggregate_formal_verdict(
        "calibration",
        [{"seed": 503, "formal_verdict": "GO"}, {"seed": 509, "formal_verdict": "GO"}],
    ) == "GO"
    assert gate.aggregate_formal_verdict(
        "calibration",
        [{"seed": 503, "formal_verdict": "GO"}, {"seed": 509, "formal_verdict": "NO-GO"}],
    ) == "NO-GO"
    with pytest.raises(ValueError, match="exact ordered result rows"):
        gate.aggregate_formal_verdict(
            "calibration", [{"seed": 503, "formal_verdict": "GO"}]
        )


def test_retinotopic_fields_are_local_across_all_orientations():
    fields = gate.retinotopic_feature_sets(8, 8, 4, 3)
    assert len(fields) == 16
    assert all(len(field) == 8 * 3 * 3 for field in fields)
    for field in fields:
        positions = {(feature // 8) % 8 * 8 + feature % 8 for feature in field}
        assert len(positions) == 9


def test_part_layer_creates_only_local_synapses_and_preserves_scramble_fanin():
    kwargs = dict(
        seed=222, n_orientations=2, n_pos=4, v2_grid=2, rf_width=2,
        part_cells=3, fs_cells=2, latency_steps=20, lp=0.05, ld_wi=0.01,
    )
    local = gate.RetinotopicPartLayer(**kwargs)
    scramble = gate.RetinotopicPartLayer(**kwargs, receptive_field_scramble=True)
    local_metrics = local.locality_metrics()
    scramble_metrics = scramble.locality_metrics()
    assert local_metrics["nonlocal_synapses"] == 0
    assert local_metrics["fan_in_preserved"] is True
    assert scramble_metrics["actual_feedforward_synapses"] == local_metrics["actual_feedforward_synapses"]
    assert scramble_metrics["nonlocal_synapses"] > 0


def test_v2_readout_returns_every_fired_cell_without_top_k_truncation(monkeypatch):
    layer = gate.RetinotopicPartLayer(
        seed=222, n_orientations=2, n_pos=4, v2_grid=2, rf_width=2,
        part_cells=4, fs_cells=2, latency_steps=40, lp=0.05, ld_wi=0.01,
        fs_enabled=False,
    )
    fired = layer.encode_all_fired(set(range(layer.n_in)))
    assert len(fired) > layer.selectors[0].k_win
    assert fired == set(range(layer.n_out))
    assert layer.selection_metrics()["host_top_k_or_first_k_truncation"] is False
    source = inspect.getsource(gate.RetinotopicPartLayer.encode_all_fired)
    assert "argsort" not in source and "argpartition" not in source


def test_both_learning_stages_use_substrate_permanences_and_trace_prior_frame():
    v2 = gate.RetinotopicPartLayer(
        seed=222, n_orientations=2, n_pos=4, v2_grid=2, rf_width=2,
        part_cells=4, fs_cells=2, latency_steps=40, lp=0.05, ld_wi=0.01,
    )
    before_v2 = v2.feedforward_permanences()
    first_v1 = {0, 1, 8, 9}
    first_v2 = v2.encode_all_fired(first_v1)
    v2.learn(first_v1, first_v2)
    assert np.count_nonzero(np.abs(v2.feedforward_permanences() - before_v2) > 1e-8) > 0

    it = gate.TraceIdentityLayer(
        seed=223, n_in=v2.n_out, n_col=16, expected_active=2,
        fs_cells=4, latency_steps=40, lp=0.05, ld_wi=0.01,
    )
    it.begin_track()
    first_it = it.encode_all_fired(first_v2)
    it.learn(first_v2, first_it, trace_enabled=True, trace_decay=0.8)
    second_v2 = set(range(v2.n_out)) - first_v2
    second_it = it.encode_all_fired(second_v2)
    it.learn(second_v2, second_it, trace_enabled=True, trace_decay=0.8)
    assert it.trace_only_update_synapses > 0
    assert it.selection_metrics()["postsynaptic_persistence_present"] is False


def test_control_set_and_no_label_learning_signatures_are_fixed():
    assert set(gate.ARM_SPECS) == {
        "intact", "all_learning_off", "v2_learning_off", "it_trace_off",
        "temporal_shuffle", "v2_local_fs_lesion", "it_fs_lesion",
        "receptive_field_scramble",
    }
    assert "label" not in inspect.signature(gate.RetinotopicPartLayer.learn).parameters
    assert "label" not in inspect.signature(gate.TraceIdentityLayer.learn).parameters
    assert "label" not in inspect.signature(gate._infer_codes).parameters


def test_scientific_diagnostics_require_both_stages_fs_and_zero_no_learning():
    base = {
        "heldout_identity_decode": 0.70,
        "held_to_train_cosine_margin": 0.15,
        "v2_learning": {"changed_synapses": 3},
        "it_learning": {"changed_synapses": 4},
        "v2_selection": {"mean_fired_fraction": 0.20},
    }
    arms = {name: dict(base) for name in gate.ARM_SPECS}
    arms["all_learning_off"] = {
        **base, "heldout_identity_decode": 0.40,
        "v2_learning": {"changed_synapses": 0}, "it_learning": {"changed_synapses": 0},
    }
    arms["v2_learning_off"] = {**base, "heldout_identity_decode": 0.50}
    arms["it_trace_off"] = {**base, "heldout_identity_decode": 0.50}
    arms["temporal_shuffle"] = {**base, "heldout_identity_decode": 0.50}
    arms["v2_local_fs_lesion"] = {
        **base, "heldout_identity_decode": 0.50,
        "v2_selection": {"mean_fired_fraction": 0.40},
    }
    arms["receptive_field_scramble"] = {**base, "heldout_identity_decode": 0.30}
    diagnostics = gate.scientific_diagnostics(
        arms, {"heldout_identity_decode": 0.30}
    )
    assert all(diagnostics.values())
    arms["all_learning_off"]["it_learning"]["changed_synapses"] = 1
    assert gate.scientific_diagnostics(
        arms, {"heldout_identity_decode": 0.30}
    )["no_learning_changes_zero"] is False


def test_tiny_smoke_is_non_scientific_and_checks_plumbing(tmp_path):
    row = gate.run_seed(gate.SMOKE_SEED, _tiny_args(tmp_path))
    assert row["phase"] == "smoke"
    assert row["scientific_evidence"] is False
    assert row["formal_verdict"] == "NOT-SCIENTIFIC-SMOKE"
    assert row["calibration_status"] == "SMOKE-ONLY"
    assert set(row["arms"]) == set(gate.ARM_SPECS)
    assert row["stream_checks"]["temporal_shuffle_preserves_exact_frame_multiset"] is True
    assert row["stream_checks"]["labels_enter_encoding_or_learning"] is False
    assert row["stream_checks"]["postsynaptic_persistence_present"] is False
    assert row["stream_checks"]["v2_host_top_k_or_first_k_truncation"] is False
    assert row["arms"]["intact"]["v2_connectivity"]["nonlocal_synapses"] == 0
    assert row["arms"]["all_learning_off"]["v2_learning"]["changed_synapses"] == 0
    assert row["arms"]["all_learning_off"]["it_learning"]["changed_synapses"] == 0
    assert row["smoke_checks"]["local_connectivity_only"] is True
    assert row["smoke_checks"]["local_v2_permanences_change"] is True
    assert row["smoke_checks"]["it_permanences_change"] is True
    assert row["smoke_checks"]["prior_frame_trace_affects_it_update"] is True
    assert row["smoke_checks"]["v2_fs_lesion_increases_density"] is True
    assert row["smoke_checks"]["labels_absent"] is True
