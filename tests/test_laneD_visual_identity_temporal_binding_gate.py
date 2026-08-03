"""Focused guards for the visual temporal-binding calibration successor."""
from __future__ import annotations

import copy
import inspect
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np
import pytest

from research.runners import _laneD_visual_identity_temporal_binding_gate as gate


def _tiny_args(tmp_path):
    return gate.build_parser().parse_args(
        [
            "--phase",
            "smoke",
            "--seeds",
            str(gate.SMOKE_SEED),
            "--epochs",
            "1",
            "--train-frames",
            "3",
            "--tracks-per-object",
            "1",
            "--held-frames",
            "2",
            "--image-size",
            "16",
            "--n-pos",
            "4",
            "--rf-radius",
            "2",
            "--v1-k-active",
            "8",
            "--v1-fs-neurons",
            "4",
            "--n-col",
            "16",
            "--k-win",
            "2",
            "--latency-fs-neurons",
            "4",
            "--latency-steps",
            "40",
            "--out",
            str(tmp_path / "unused.json"),
        ]
    )


def _small_pooler(seed=gate.SMOKE_SEED, persistence_gain=3.0):
    return gate.TemporalBindingPooler(
        seed=seed,
        n_in=8,
        n_col=16,
        k_win=2,
        lp=0.05,
        ld_wi=0.01,
        homeostasis_rate=0.01,
        homeostasis_strength=6.0,
        persistence_decay=0.72,
        persistence_gain=persistence_gain,
        latency_fs_neurons=4,
        latency_steps=40,
    )


def test_seed_partitions_are_fresh_disjoint_and_smoke_is_reserved():
    partitions = [set(values) for values in gate.SEED_PARTITIONS.values()]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    prior = {seed for values in gate.PRIOR_SEED_PARTITIONS.values() for seed in values}
    assert set().union(*partitions).isdisjoint(prior)
    assert gate.SMOKE_SEED not in set().union(*partitions)
    assert gate.build_parser().parse_args([]).seeds == [224, 225]
    gate.validate_seed_partition("smoke", [gate.SMOKE_SEED])
    gate.validate_seed_partition("calibration", [224, 225])
    gate.validate_individual_seed("calibration", 224)
    gate.validate_individual_seed("calibration", 225)
    with pytest.raises(ValueError, match="exact ordered seeds"):
        gate.validate_seed_partition("calibration", [224])
    with pytest.raises(ValueError, match="exact ordered seeds"):
        gate.validate_seed_partition("calibration", [224, 224])
    with pytest.raises(ValueError, match="exact ordered seeds"):
        gate.validate_seed_partition("calibration", [225, 224])
    with pytest.raises(ValueError, match="exact ordered seeds"):
        gate.validate_seed_partition("calibration", [gate.SMOKE_SEED])
    with pytest.raises(ValueError, match="exact ordered seeds"):
        gate.validate_seed_partition("smoke", [224])
    with pytest.raises(ValueError, match="is not open"):
        gate.validate_seed_partition("development", [226])
    with pytest.raises(ValueError, match="is not open"):
        gate.validate_seed_partition("heldout", [323])
    assert gate.build_parser()._option_string_actions["--phase"].choices == (
        "smoke",
        "calibration",
    )


def test_formal_verdict_helpers_keep_smoke_separate_and_require_both_seeds():
    assert gate.formal_verdict_for_phase("smoke") == "NOT-SCIENTIFIC-SMOKE"
    assert gate.formal_verdict_for_phase("calibration", "GO") == "GO"
    with pytest.raises(ValueError, match="earned GO"):
        gate.formal_verdict_for_phase("calibration")
    assert gate.aggregate_formal_verdict("smoke", []) == "NOT-SCIENTIFIC-SMOKE"
    assert gate.aggregate_formal_verdict(
        "calibration",
        [
            {"seed": 224, "formal_verdict": "GO"},
            {"seed": 225, "formal_verdict": "GO"},
        ],
    ) == "GO"
    assert gate.aggregate_formal_verdict(
        "calibration",
        [
            {"seed": 224, "formal_verdict": "GO"},
            {"seed": 225, "formal_verdict": "NO-GO"},
        ],
    ) == "NO-GO"
    assert gate.aggregate_formal_verdict(
        "calibration",
        [
            {"seed": 224, "formal_verdict": "GO"},
            {"seed": 225, "formal_verdict": "UNDEFINED"},
        ],
    ) == "UNDEFINED"
    for invalid_rows in (
        [{"seed": 224, "formal_verdict": "GO"}],
        [
            {"seed": 224, "formal_verdict": "GO"},
            {"seed": 224, "formal_verdict": "GO"},
        ],
        [
            {"seed": 225, "formal_verdict": "GO"},
            {"seed": 224, "formal_verdict": "GO"},
        ],
    ):
        with pytest.raises(ValueError, match="exact ordered result rows"):
            gate.aggregate_formal_verdict("calibration", invalid_rows)


def test_run_seed_rejects_smoke_seed_as_calibration_before_building(monkeypatch, tmp_path):
    args = _tiny_args(tmp_path)
    args.phase = "calibration"

    def fail_dataset_build(*_args, **_kwargs):
        raise AssertionError("dataset construction happened before seed validation")

    monkeypatch.setattr(gate, "build_visual_dataset", fail_dataset_build)
    with pytest.raises(ValueError, match="cannot use seed"):
        gate.run_seed(gate.SMOKE_SEED, args)


def test_diagnostics_require_learning_fs_and_strict_scramble_controls():
    arms = {
        "intact": {
            "heldout_identity_decode": 0.75,
            "held_to_train_cosine_margin": 0.40,
            "usage": {"usage_cv": 0.20},
            "local_learning": {"changed_synapses": 3},
        },
        "temporal_shuffle": {"held_to_train_cosine_margin": 0.20},
        "persistence_lesion": {"held_to_train_cosine_margin": 0.20},
        "trace_lesion": {"held_to_train_cosine_margin": 0.20},
        "homeostasis_lesion": {"usage": {"usage_cv": 0.30}},
        "no_learning": {
            "heldout_identity_decode": 0.50,
            "held_to_train_cosine_margin": 0.10,
            "local_learning": {"changed_synapses": 0},
        },
    }
    selection = {
        "graded_winners": [0, 1],
        "neural_drive_lesion_winners": [],
        "fs_intact_columns_fired": 2,
        "fs_lesion_columns_fired": 3,
        "graded_host_reference_overlap": 1.0,
        "flat_drive_overlap_with_graded": 0.0,
    }

    diagnostics = gate.calibration_diagnostics(
        arms,
        {"heldout_identity_decode": 0.25},
        selection,
        selection,
        v1_k_active=2,
        k_win=2,
        n_col=16,
    )
    assert all(diagnostics.values())

    def evaluate(
        trial_arms=arms,
        pixel_decode=0.25,
        v1_selection=selection,
        identity_selection=selection,
    ):
        return gate.calibration_diagnostics(
            trial_arms,
            {"heldout_identity_decode": pixel_decode},
            v1_selection,
            identity_selection,
            v1_k_active=2,
            k_win=2,
            n_col=16,
        )

    trial_arms = copy.deepcopy(arms)
    trial_arms["no_learning"]["heldout_identity_decode"] = 0.51
    assert evaluate(trial_arms)["no_learning_costs_decode"] is False

    trial_arms = copy.deepcopy(arms)
    trial_arms["no_learning"]["held_to_train_cosine_margin"] = 0.40
    assert evaluate(trial_arms)["no_learning_costs_margin"] is False

    trial_arms = copy.deepcopy(arms)
    trial_arms["no_learning"]["local_learning"]["changed_synapses"] = 1
    assert evaluate(trial_arms)["no_learning_changes_no_permanences"] is False

    fs_lesion_inactive = {**selection, "fs_lesion_columns_fired": 2}
    assert evaluate(v1_selection=fs_lesion_inactive)["v1_fs_pathway_is_causal"] is False
    assert (
        evaluate(identity_selection=fs_lesion_inactive)[
            "identity_fs_pathway_is_causal"
        ]
        is False
    )
    assert evaluate(pixel_decode=0.26)["pixel_scramble_at_or_below_chance"] is False
    assert evaluate(pixel_decode=0.51)["pixel_scramble_costs_decode"] is False


def test_v1_encoder_selects_from_spikes_without_host_activation_top_k(monkeypatch):
    encoder = gate.SpikingV1Encoder(
        seed=gate.SMOKE_SEED,
        n_features=16,
        k_active=2,
        n_fs=4,
        n_steps=40,
    )

    def fail_host_ranking(*_args, **_kwargs):
        raise AssertionError("host activation top-k was called")

    monkeypatch.setattr(gate.np, "argsort", fail_host_ranking)
    monkeypatch.setattr(gate.np, "argpartition", fail_host_ranking)
    winners = encoder.encode(np.linspace(0.2, 6.0, 16))

    assert winners == {14, 15}
    assert encoder.selector.last_fired_count >= 2
    source = inspect.getsource(gate.SpikingV1Encoder.encode)
    assert "selector.select" in source
    assert "argsort" not in source and "argpartition" not in source


def test_identity_winners_use_spikes_and_persistence_is_lesionable(monkeypatch):
    pooler = _small_pooler(persistence_gain=20.0)
    persistence = np.zeros(16, dtype=np.float64)
    persistence[[3, 9]] = 1.0

    def fail_host_ranking(*_args, **_kwargs):
        raise AssertionError("host identity top-k was called")

    monkeypatch.setattr(gate.np, "argsort", fail_host_ranking)
    monkeypatch.setattr(gate.np, "argpartition", fail_host_ranking)
    biased = pooler._winners(set(), persistence=persistence)
    lesioned = pooler._winners(set(), persistence=None)

    assert biased == {3, 9}
    assert lesioned == set()
    assert pooler.persistence_bias_presentations == 1
    assert "label" not in inspect.signature(gate.infer_codes).parameters
    assert "object" not in inspect.signature(gate.infer_codes).parameters
    source = inspect.getsource(gate.TemporalBindingPooler._winners)
    assert "latency_selector.select" in source
    assert "argsort" not in source and "argpartition" not in source


def test_local_pre_post_learning_changes_substrate_permanences():
    pooler = _small_pooler()
    before = pooler.feedforward_permanences()
    tracks = (({0, 1, 2}, {1, 2, 3}, {2, 3, 4}),)
    pooler.train_tracks(
        tracks,
        epochs=2,
        trace_decay=0.68,
        trace_enabled=True,
        persistence_enabled=True,
        homeostasis_mode="slow",
        seed=gate.SMOKE_SEED,
    )
    after = pooler.feedforward_permanences()

    assert np.count_nonzero(np.abs(after - before) > 1e-8) > 0
    assert pooler.persistence_bias_presentations > 0
    assert pooler.latency_selector.selection_calls == 6
    assert pooler.feedforward_permanences().shape == (8 * 16,)


def test_controls_keep_temporal_and_mechanistic_lesions_distinct():
    assert set(gate.ARM_SPECS) == {
        "intact",
        "temporal_shuffle",
        "persistence_lesion",
        "trace_lesion",
        "homeostasis_lesion",
        "no_learning",
    }
    assert gate.ARM_SPECS["temporal_shuffle"]["shuffle"] is True
    assert gate.ARM_SPECS["persistence_lesion"]["persistence"] is False
    assert gate.ARM_SPECS["trace_lesion"]["trace"] is False
    assert gate.ARM_SPECS["homeostasis_lesion"]["homeostasis"] == "none"
    assert gate.ARM_SPECS["no_learning"]["learning"] is False


def test_tiny_cpu_smoke_builds_controls_and_machine_preconditions(tmp_path):
    row = gate.run_seed(gate.SMOKE_SEED, _tiny_args(tmp_path))

    assert set(row["arms"]) == set(gate.ARM_SPECS)
    assert row["phase"] == "smoke"
    assert row["scientific_evidence"] is False
    assert row["calibration_status"] == "SMOKE-ONLY"
    assert row["calibration_readiness_verdict"] == "NOT-SCIENTIFIC-SMOKE"
    assert row["formal_verdict"] == "NOT-SCIENTIFIC-SMOKE"
    assert row["stream_checks"]["fresh_seed_partitions"] is True
    assert row["stream_checks"]["smoke_seed_outside_scientific_partitions"] is True
    assert row["stream_checks"]["same_multiset_after_temporal_shuffle"] is True
    assert row["stream_checks"]["labels_enter_learning_or_inference"] is False
    assert row["stream_checks"]["host_top_k_determines_v1_spikes"] is False
    assert row["stream_checks"]["host_top_k_determines_identity_winners"] is False
    assert row["v1_selection_telemetry"]["selection_route"] == "first_spike_latency"
    assert row["identity_selection_telemetry"]["selection_route"] == "first_spike_latency"
    assert row["v1_selection"]["neural_drive_lesion_winners"] == []
    assert row["identity_selection"]["neural_drive_lesion_winners"] == []
    assert len(row["v1_selection"]["graded_winners"]) == 8
    assert len(row["identity_selection"]["graded_winners"]) == 2
    assert (
        row["v1_selection"]["fs_lesion_columns_fired"]
        > row["v1_selection"]["fs_intact_columns_fired"]
    )
    assert (
        row["identity_selection"]["fs_lesion_columns_fired"]
        > row["identity_selection"]["fs_intact_columns_fired"]
    )
    assert row["arms"]["intact"]["local_learning"]["changed_synapses"] > 0
    assert row["arms"]["no_learning"]["local_learning"]["changed_synapses"] == 0
    assert row["preconditions"]
    assert all(check["ok"] for check in row["preconditions"])
    for arm in row["arms"].values():
        assert 0.0 <= arm["heldout_identity_decode"] <= 1.0
        assert arm["mean_code_activity"] == pytest.approx(2 / 16, abs=1e-4)
    assert 0.0 <= row["pixel_scramble"]["heldout_identity_decode"] <= 1.0
