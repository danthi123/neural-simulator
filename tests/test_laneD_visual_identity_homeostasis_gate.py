"""Focused guards for the bounded visual identity/homeostasis calibration gate."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from research.runners import _laneD_visual_identity_homeostasis_gate as gate


def test_seed_partitions_are_disjoint_and_calibration_is_default():
    partitions = [set(values) for values in gate.SEED_PARTITIONS.values()]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    assert gate.build_parser().parse_args([]).seeds == [212, 213]
    gate.validate_seed_partition("calibration", [212, 213])
    with pytest.raises(ValueError, match="cannot use seeds"):
        gate.validate_seed_partition("calibration", [214])


def test_dataset_has_four_continuous_objects_and_disjoint_held_transforms():
    dataset = gate.build_visual_dataset(
        seed=212,
        image_size=16,
        train_frames=5,
        tracks_per_object=1,
        held_frames=3,
        pixel_noise=0.0,
    )
    assert set(dataset.train_object_ids) == {0, 1, 2, 3}
    assert set(dataset.held_object_ids) == {0, 1, 2, 3}
    assert len(dataset.train_tracks) == 4
    assert all(len(track) == 5 for track in dataset.train_tracks)
    for track in dataset.train_tracks:
        assert len(set(dataset.train_object_ids[list(track)])) == 1
        transforms = [dataset.train_transforms[index] for index in track]
        adjacent = [
            abs(left.x - right.x) + abs(left.y - right.y)
            for left, right in zip(transforms, transforms[1:])
        ]
        assert max(adjacent) < 0.35
    train_transform_set = set(dataset.train_transforms)
    assert train_transform_set.isdisjoint(set(dataset.held_transforms))
    assert dataset.train_images.shape[1:] == (2, 16, 16)
    assert not np.allclose(dataset.train_images[0], dataset.train_images[5])


def test_temporal_shuffle_preserves_frames_but_breaks_order():
    tracks = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
    shuffled = gate.shuffled_track_indices(tracks, seed=212)
    assert [len(track) for track in shuffled] == [4, 4, 4]
    assert sorted(index for track in shuffled for index in track) == list(range(12))
    assert shuffled != tracks


def test_slow_homeostasis_suppresses_overused_columns_without_labels():
    usage = np.full(6, 2 / 6, dtype=np.float64)
    for _ in range(20):
        usage = gate.update_usage_ema(usage, (0, 1), rate=0.01)
    gain = gate.homeostatic_gain(usage, target=2 / 6, strength=6.0)
    assert usage[0] > usage[2]
    assert gain[0] < gain[2]
    assert 0.20 <= gain.min() <= gain.max() <= 5.0
    assert "label" not in inspect.signature(gate.infer_codes).parameters
    assert "object" not in inspect.signature(gate.infer_codes).parameters


def test_latency_selector_uses_spikes_resets_state_and_obeys_neural_lesions():
    selector = gate.SpikeLatencySelector(
        seed=212,
        n_col=16,
        k_win=2,
        n_fs=8,
        n_steps=40,
    )
    graded = np.linspace(0.2, 6.0, 16)
    first = selector.select(graded)
    first_latency = selector.last_first_spike.copy()
    first_fired = selector.last_fired_count
    repeated = selector.select(graded)

    assert first == repeated == {14, 15}
    assert np.array_equal(first_latency, selector.last_first_spike)
    assert first_fired == selector.last_fired_count >= 2
    assert max(first_latency[list(first)]) < min(first_latency[:12])

    flat = selector.select(np.full(16, graded.mean()))
    assert len(flat) == 2
    assert len(flat & first) / 2 <= 2 / 16 + 0.25
    assert selector.select(graded, neural_drive_enabled=False) == set()
    assert selector.last_fired_count == 0


def test_fs_lesion_preserves_latency_winners_but_releases_later_spiking():
    graded = np.linspace(0.2, 6.0, 16)
    intact = gate.SpikeLatencySelector(212, 16, 2, 8, 40, wta_enabled=True)
    lesioned = gate.SpikeLatencySelector(212, 16, 2, 8, 40, wta_enabled=False)
    intact_winners = intact.select(graded)
    lesioned_winners = lesioned.select(graded)

    assert intact_winners == lesioned_winners == {14, 15}
    assert lesioned.last_fired_count > intact.last_fired_count


def test_pooler_winner_selection_never_calls_host_drive_top_k(monkeypatch):
    pooler = gate.HomeostaticTracePooler(
        seed=212,
        n_in=8,
        n_col=16,
        k_win=2,
        lp=0.05,
        ld_wi=0.01,
        homeostasis_rate=0.01,
        homeostasis_strength=6.0,
        latency_fs_neurons=8,
        latency_steps=40,
    )

    def fail_host_ranking(*_args, **_kwargs):
        raise AssertionError("host drive top-k was called")

    monkeypatch.setattr(gate.np, "argsort", fail_host_ranking)
    monkeypatch.setattr(gate.np, "argpartition", fail_host_ranking)
    winners = pooler._winners({0, 1, 2, 3})
    assert len(winners) == 2
    assert pooler.latency_selector.selection_calls == 1
    source = inspect.getsource(gate.HomeostaticTracePooler._winners)
    assert "latency_selector.select" in source
    assert "argsort" not in source and "argpartition" not in source


def test_controls_are_mechanistically_distinct():
    assert set(gate.ARM_SPECS) == {
        "intact",
        "trace_only",
        "temporal_shuffle",
        "no_learning",
        "trace_lesion",
        "homeostasis_lesion",
    }
    assert gate.ARM_SPECS["trace_only"]["homeostasis"] == "epoch"
    assert gate.ARM_SPECS["homeostasis_lesion"]["homeostasis"] == "none"
    assert gate.ARM_SPECS["trace_lesion"]["trace"] is False
    assert gate.ARM_SPECS["temporal_shuffle"]["shuffle"] is True


def test_tiny_cpu_smoke_builds_every_control(tmp_path):
    args = gate.build_parser().parse_args(
        [
            "--phase", "calibration",
            "--seeds", "212",
            "--epochs", "1",
            "--train-frames", "3",
            "--tracks-per-object", "1",
            "--held-frames", "2",
            "--image-size", "16",
            "--n-pos", "4",
            "--rf-radius", "2",
            "--n-col", "16",
            "--k-win", "2",
            "--t-active", "8",
            "--out", str(tmp_path / "unused.json"),
        ]
    )
    row = gate.run_seed(212, args)
    assert set(row["arms"]) == set(gate.ARM_SPECS)
    assert row["formal_verdict"] == "NOT-RUN-CALIBRATION-ONLY"
    assert row["stream_checks"]["same_multiset_after_temporal_shuffle"] is True
    assert row["stream_checks"]["labels_enter_training_or_inference"] is False
    assert row["stream_checks"]["host_top_k_determines_pooler_winners"] is False
    assert row["selection_telemetry"]["selection_route"] == "first_spike_latency"
    assert row["selection_telemetry"]["host_drive_ranking_used_for_winners"] is False
    assert row["neural_selection"]["neural_drive_lesion_winners"] == []
    assert len(row["neural_selection"]["graded_winners"]) == 2
    assert row["n_objects"] == 4
    for arm in row["arms"].values():
        assert 0.0 <= arm["heldout_identity_decode"] <= 1.0
        assert arm["mean_code_activity"] == pytest.approx(2 / 16, abs=1e-4)
    assert 0.0 <= row["pixel_scramble"]["heldout_identity_decode"] <= 1.0
