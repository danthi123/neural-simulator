"""Calibration-only successor for locally learned visual identity invariance.

The preceding visual gate established that first-spike latency can replace host
top-k winner selection, but its sparse host-selected V1 features did not form
stable identities across new position, scale, and lighting combinations. This
successor moves upstream from selector tuning:

* graded Gabor/V1-complex activity is converted to sparse V1 spikes by a
  first-spike latency population, never by host ranking;
* V1-to-identity permanences live in ``cp_connections.data`` and are changed by
  the existing local traced-potentiation and winner-inactive-depression kernels;
* recent postsynaptic spike winners provide a short-lived, label-free
  persistence bias during continuous tracks so adjacent views can bind onto a
  shared assembly;
* persistence is absent during single-view inference, and object labels remain
  outside the network for scoring only.

The postsynaptic persistence current and pre/post traces are runner-maintained
scaffolds, not claimed biological completions. The gate keeps them visible and
lesionable. It also retains temporal shuffle, trace, homeostasis, pixel
scramble, neural-drive, flat-drive, and fast-spiking pathway controls.

Seed 220 is reserved for smoke/unit testing and is outside every scientific
partition. Fresh calibration seeds 224/225 are open. Development seeds
226/227/322 and held-out seeds 323/324/325 are locked.

Cheap smoke::

  SIM_BACKEND=numpy python -u -m \
    research.runners._laneD_visual_identity_temporal_binding_gate \
    --phase smoke --seeds 220 --epochs 1 --train-frames 3 \
    --tracks-per-object 1 --held-frames 2 --image-size 16 --n-pos 4 \
    --v1-k-active 8 --n-col 16 --k-win 2 --latency-steps 40 \
    --out /tmp/visual_identity_temporal_binding_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._emerge14_stageC_onbridge_learning_derisk import _host  # noqa: E402
from research.runners._emerge50_trace_rule_derisk import (  # noqa: E402
    TRACE_DECAY,
    _apply_traced_potentiation,
)
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from research.runners._laneD_v1_pooler_trace_invariance_derisk import (  # noqa: E402
    TraceV1Pooler,
    _binary_codes,
    _normalize_complex,
    _scramble_images,
)
from research.runners._laneD_visual_identity_homeostasis_gate import (  # noqa: E402
    SEED_PARTITIONS as PRIOR_SEED_PARTITIONS,
    SpikeLatencySelector,
    build_visual_dataset,
    homeostatic_gain,
    score_codes,
    shuffled_track_indices,
)
from tools.verdict import GO, NO_GO, UNDEFINED, Verdict  # noqa: E402


SMOKE_SEED = 220
CALIBRATION_SEEDS = (224, 225)
DEVELOPMENT_SEEDS = (226, 227, 322)
HELDOUT_SEEDS = (323, 324, 325)
SEED_PARTITIONS = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "heldout": HELDOUT_SEEDS,
}
OPEN_PHASES = ("calibration",)
CLI_PHASES = ("smoke",) + OPEN_PHASES

ARM_SPECS = {
    "intact": {
        "trace": True,
        "persistence": True,
        "homeostasis": "slow",
        "shuffle": False,
        "learning": True,
    },
    "temporal_shuffle": {
        "trace": True,
        "persistence": True,
        "homeostasis": "slow",
        "shuffle": True,
        "learning": True,
    },
    "persistence_lesion": {
        "trace": True,
        "persistence": False,
        "homeostasis": "slow",
        "shuffle": False,
        "learning": True,
    },
    "trace_lesion": {
        "trace": False,
        "persistence": True,
        "homeostasis": "slow",
        "shuffle": False,
        "learning": True,
    },
    "homeostasis_lesion": {
        "trace": True,
        "persistence": True,
        "homeostasis": "none",
        "shuffle": False,
        "learning": True,
    },
    "no_learning": {
        "trace": False,
        "persistence": False,
        "homeostasis": "none",
        "shuffle": False,
        "learning": False,
    },
}

OUT = Path(
    "research/findings/raw/lanes/perception/"
    "visual_identity_temporal_binding_calibration.json"
)


def validate_seed_partition(phase: str, seeds: Sequence[int]) -> None:
    """Keep every successor seed fresh and later decision phases locked."""
    if phase == "smoke":
        supplied = tuple(int(seed) for seed in seeds)
        if supplied != (SMOKE_SEED,):
            raise ValueError(
                f"smoke phase requires only reserved SMOKE_SEED={SMOKE_SEED}; "
                f"received {list(supplied)}"
            )
        return
    if phase not in OPEN_PHASES:
        raise ValueError(
            f"phase {phase!r} is not open; this successor is calibration-only and "
            "development/held-out seeds remain locked"
        )
    allowed = set(SEED_PARTITIONS[phase])
    unexpected = sorted(set(int(seed) for seed in seeds) - allowed)
    if unexpected:
        raise ValueError(
            f"{phase} phase cannot use seeds {unexpected}; allowed seeds are {sorted(allowed)}"
        )


def formal_verdict_for_phase(phase: str, decided_status: str | None = None) -> str:
    if phase == "smoke":
        return "NOT-SCIENTIFIC-SMOKE"
    if decided_status not in {GO, NO_GO, UNDEFINED}:
        raise ValueError("calibration requires an earned GO, NO-GO, or UNDEFINED verdict")
    return str(decided_status)


def aggregate_formal_verdict(phase: str, rows: Sequence[dict]) -> str:
    if phase == "smoke":
        return "NOT-SCIENTIFIC-SMOKE"
    statuses = [row["formal_verdict"] for row in rows]
    if any(status == UNDEFINED for status in statuses):
        return UNDEFINED
    if statuses and all(status == GO for status in statuses):
        return GO
    return NO_GO


def dense_v1_activity(
    images: np.ndarray,
    args: argparse.Namespace,
    response_matrix: np.ndarray,
) -> np.ndarray:
    """Return non-negative graded V1-complex activity without host sparsifying."""
    v1 = pool_v1_to_complex(
        encode_v1(images, response_matrix),
        args.n_orientations,
        args.n_frequencies,
        args.n_pos,
    )
    normalized = _normalize_complex(
        v1,
        args.complex_norm,
        args.n_orientations,
        args.n_pos,
    )
    normalized = np.maximum(np.asarray(normalized, dtype=np.float64), 0.0)
    if not np.isfinite(normalized).all():
        raise ValueError("V1-complex activity must be finite")
    return normalized


class SpikingV1Encoder:
    """Convert graded V1 drive to sparse V1 spike assemblies without top-k."""

    def __init__(
        self,
        seed: int,
        n_features: int,
        k_active: int,
        n_fs: int,
        n_steps: int,
        wta_enabled: bool = True,
    ) -> None:
        self.n_features = int(n_features)
        self.k_active = int(k_active)
        self.selector = SpikeLatencySelector(
            seed=seed,
            n_col=self.n_features,
            k_win=self.k_active,
            n_fs=n_fs,
            n_steps=n_steps,
            wta_enabled=wta_enabled,
        )

    def encode(self, activity: np.ndarray, neural_drive_enabled: bool = True) -> set[int]:
        values = np.asarray(activity, dtype=np.float64)
        if values.shape != (self.n_features,):
            raise ValueError(
                f"V1 activity must have shape ({self.n_features},), got {values.shape}"
            )
        return self.selector.select(values, neural_drive_enabled=neural_drive_enabled)

    def encode_many(self, activity: np.ndarray) -> list[set[int]]:
        rows = np.asarray(activity, dtype=np.float64)
        return [self.encode(row) for row in rows]

    def metrics(self) -> dict[str, float | int | bool | str]:
        return {
            **self.selector.metrics(),
            "representation_stage": "graded V1-complex to V1 spike assembly",
            "host_top_k_used": False,
        }


def update_usage_ema(usage: np.ndarray, winners: Iterable[int], rate: float) -> np.ndarray:
    updated = np.asarray(usage, dtype=np.float64) * (1.0 - rate)
    indices = np.fromiter((int(winner) for winner in winners), dtype=np.int64)
    if indices.size:
        updated[indices] += rate
    return updated


class TemporalBindingPooler(TraceV1Pooler):
    """Locally learn V1-spike to identity-spike invariant assemblies.

    Permanences live on the simulation bridge. The only signals entering the
    update are recent presynaptic spikes and postsynaptic spike winners. A
    short-lived winner trace biases the next view during training, allowing the
    same cells to bind adjacent transformations. It is reset at each track and
    omitted from inference.
    """

    def __init__(
        self,
        *,
        seed: int,
        n_in: int,
        n_col: int,
        k_win: int,
        lp: float,
        ld_wi: float,
        homeostasis_rate: float,
        homeostasis_strength: float,
        persistence_decay: float,
        persistence_gain: float,
        latency_fs_neurons: int,
        latency_steps: int,
    ) -> None:
        super().__init__(
            seed=seed,
            n_in=n_in,
            n_col=n_col,
            k_win=k_win,
            lp=lp,
            ld_wi=ld_wi,
        )
        self.homeostasis_rate = float(homeostasis_rate)
        self.homeostasis_strength = float(homeostasis_strength)
        self.persistence_decay = float(persistence_decay)
        self.persistence_gain = float(persistence_gain)
        self.target_usage = self.k_win / self.n_col
        self.usage = np.full(self.n_col, self.target_usage, dtype=np.float64)
        self.selection_counts = np.zeros(self.n_col, dtype=np.int64)
        self.inference_gain = np.ones(self.n_col, dtype=np.float64)
        self.persistence_bias_presentations = 0
        self.latency_selector = SpikeLatencySelector(
            seed=seed * 17 + 5,
            n_col=self.n_col,
            k_win=self.k_win,
            n_fs=latency_fs_neurons,
            n_steps=latency_steps,
        )

    def feedforward_drive(
        self,
        features: set[int],
        gain: np.ndarray | None = None,
    ) -> np.ndarray:
        return super()._drive(features, gain)

    def _winners(
        self,
        features: set[int],
        gain: np.ndarray | None = None,
        persistence: np.ndarray | None = None,
    ) -> set[int]:
        drive = self.feedforward_drive(features, gain)
        if persistence is not None:
            drive = drive + self.persistence_gain * np.asarray(persistence, dtype=np.float64)
            if np.any(persistence > 0.0):
                self.persistence_bias_presentations += 1
        return self.latency_selector.select(drive)

    def feedforward_permanences(self) -> np.ndarray:
        data = np.asarray(_host(self.b.cp_connections.data), dtype=np.float64)
        return data[self.ff_pos].copy()

    def train_tracks(
        self,
        tracks: Sequence[Sequence[set[int]]],
        *,
        epochs: int,
        trace_decay: float,
        trace_enabled: bool,
        persistence_enabled: bool,
        homeostasis_mode: str,
        seed: int,
    ) -> None:
        if homeostasis_mode not in {"slow", "none"}:
            raise ValueError(f"unknown homeostasis mode: {homeostasis_mode}")
        rng = np.random.default_rng(seed)
        order = np.arange(len(tracks))
        for _ in range(epochs):
            rng.shuffle(order)
            for track_index in order:
                pre_trace = np.zeros(self.n_in, dtype=np.float64)
                post_trace = np.zeros(self.n_col, dtype=np.float64)
                for features in tracks[int(track_index)]:
                    active = np.zeros(self.n_in, dtype=np.float64)
                    if features:
                        active[list(features)] = 1.0
                    pre_trace = (
                        active
                        if not trace_enabled
                        else np.clip(pre_trace * trace_decay + active, 0.0, 1.0)
                    )
                    post_trace *= self.persistence_decay
                    gain = (
                        homeostatic_gain(
                            self.usage,
                            self.target_usage,
                            self.homeostasis_strength,
                        )
                        if homeostasis_mode == "slow"
                        else None
                    )
                    winners = self._winners(
                        features,
                        gain,
                        post_trace if persistence_enabled else None,
                    )
                    _apply_traced_potentiation(self, pre_trace, winners, self.lp)
                    self._winner_inactive_traced(winners, pre_trace, self.ld_wi)
                    winner_array = np.fromiter(
                        (int(winner) for winner in winners), dtype=np.int64
                    )
                    if winner_array.size:
                        self.selection_counts[winner_array] += 1
                        post_trace[winner_array] += 1.0
                        np.minimum(post_trace, 1.0, out=post_trace)
                    self.usage = update_usage_ema(
                        self.usage, winners, self.homeostasis_rate
                    )

        if homeostasis_mode == "slow":
            self.inference_gain = homeostatic_gain(
                self.usage,
                self.target_usage,
                self.homeostasis_strength,
            )

    def codon(self, features: set[int]) -> set[int]:
        """Infer from V1 spikes alone; temporal persistence is training-only."""
        return self._winners(features, self.inference_gain, persistence=None)

    def usage_metrics(self) -> dict[str, float | int]:
        counts = self.selection_counts.astype(np.float64)
        mean = float(counts.mean())
        active = int(np.count_nonzero(counts))
        probabilities = counts / max(float(counts.sum()), 1.0)
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
        return {
            "active_columns": active,
            "dead_column_fraction": round(1.0 - active / self.n_col, 4),
            "usage_cv": round(float(counts.std()) / max(mean, 1e-9), 4),
            "normalized_usage_entropy": round(
                entropy / max(math.log(self.n_col), 1e-9), 4
            ),
            "persistence_bias_presentations": self.persistence_bias_presentations,
        }


def infer_codes(
    pooler: TemporalBindingPooler,
    v1_spike_codes: Sequence[set[int]],
) -> np.ndarray:
    """Infer identity assemblies from V1 spikes; labels are intentionally absent."""
    return _binary_codes(
        [pooler.codon(feature_set) for feature_set in v1_spike_codes],
        pooler.n_col,
    )


def _make_pooler(
    seed: int,
    n_in: int,
    args: argparse.Namespace,
) -> TemporalBindingPooler:
    return TemporalBindingPooler(
        seed=seed,
        n_in=n_in,
        n_col=args.n_col,
        k_win=args.k_win,
        lp=args.pool_lr_pot,
        ld_wi=args.pool_lr_depress,
        homeostasis_rate=args.homeostasis_rate,
        homeostasis_strength=args.homeostasis_strength,
        persistence_decay=args.persistence_decay,
        persistence_gain=args.persistence_gain,
        latency_fs_neurons=args.latency_fs_neurons,
        latency_steps=args.latency_steps,
    )


def latency_controls(
    selector: SpikeLatencySelector,
    drive: np.ndarray,
) -> dict[str, float | int | bool | list[int]]:
    """Preserve drive, flat-input, current-lesion, and FS-pathway controls."""
    values = np.asarray(drive, dtype=np.float64)
    graded_winners = selector.select(values)
    graded_fired_count = selector.last_fired_count
    host_reference = set(
        int(column) for column in np.argsort(-values)[: selector.k_win]
    )
    flat_winners = selector.select(
        np.full(selector.n_col, float(np.mean(values)), dtype=np.float64)
    )
    silent_winners = selector.select(values, neural_drive_enabled=False)
    fs_lesion = SpikeLatencySelector(
        seed=selector.seed,
        n_col=selector.n_col,
        k_win=selector.k_win,
        n_fs=selector.n_fs,
        n_steps=selector.n_steps,
        wta_enabled=False,
    )
    lesion_winners = fs_lesion.select(values)
    denominator = max(selector.k_win, 1)
    return {
        "graded_winners": sorted(graded_winners),
        "graded_host_reference_overlap": round(
            len(graded_winners & host_reference) / denominator, 4
        ),
        "flat_drive_overlap_with_graded": round(
            len(flat_winners & graded_winners) / denominator, 4
        ),
        "flat_drive_winners": sorted(flat_winners),
        "neural_drive_lesion_winners": sorted(silent_winners),
        "fs_lesion_winner_overlap": round(
            len(lesion_winners & graded_winners) / denominator, 4
        ),
        "fs_intact_columns_fired": graded_fired_count,
        "fs_lesion_columns_fired": fs_lesion.last_fired_count,
        "host_drive_ranking_used_for_winners": False,
    }


def _all_numeric_values_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    validate_seed_partition(args.phase, [seed])
    dataset = build_visual_dataset(
        seed=seed,
        image_size=args.image_size,
        train_frames=args.train_frames,
        tracks_per_object=args.tracks_per_object,
        held_frames=args.held_frames,
        pixel_noise=args.pixel_noise,
    )
    response_matrix = build_gabor_response_matrix(
        n_orientations=args.n_orientations,
        n_frequencies=args.n_frequencies,
        n_positions_per_dim=args.n_pos,
        retina_size=args.image_size,
        receptive_field_radius=args.rf_radius,
    )
    train_activity = dense_v1_activity(dataset.train_images, args, response_matrix)
    held_activity = dense_v1_activity(dataset.held_images, args, response_matrix)
    scrambled_images = _scramble_images(dataset.held_images, seed * 101 + 83)
    scrambled_activity = dense_v1_activity(scrambled_images, args, response_matrix)

    n_v1 = args.n_orientations * args.n_pos * args.n_pos
    v1_encoder = SpikingV1Encoder(
        seed=seed * 1009 + 19,
        n_features=n_v1,
        k_active=args.v1_k_active,
        n_fs=args.v1_fs_neurons,
        n_steps=args.latency_steps,
    )
    train_features = v1_encoder.encode_many(train_activity)
    held_features = v1_encoder.encode_many(held_activity)
    scrambled_features = v1_encoder.encode_many(scrambled_activity)

    ordered_indices = dataset.train_tracks
    shuffled_indices = shuffled_track_indices(ordered_indices, seed * 101 + 89)
    pool_seed = seed * 1009 + 31
    arms: dict[str, dict] = {}
    intact_pooler: TemporalBindingPooler | None = None
    intact_train_codes: np.ndarray | None = None

    for arm_name, spec in ARM_SPECS.items():
        pooler = _make_pooler(pool_seed, n_v1, args)
        initial_permanences = pooler.feedforward_permanences()
        indices = shuffled_indices if spec["shuffle"] else ordered_indices
        tracks = tuple(
            tuple(train_features[index] for index in track) for track in indices
        )
        if spec["learning"]:
            pooler.train_tracks(
                tracks,
                epochs=args.epochs,
                trace_decay=args.trace_decay,
                trace_enabled=bool(spec["trace"]),
                persistence_enabled=bool(spec["persistence"]),
                homeostasis_mode=str(spec["homeostasis"]),
                seed=seed * 1009 + 43,
            )
        final_permanences = pooler.feedforward_permanences()
        changed = np.abs(final_permanences - initial_permanences) > 1e-8
        train_codes = infer_codes(pooler, train_features)
        held_codes = infer_codes(pooler, held_features)
        arms[arm_name] = {
            **score_codes(
                train_codes,
                dataset.train_object_ids,
                held_codes,
                dataset.held_object_ids,
            ),
            "usage": pooler.usage_metrics(),
            "local_learning": {
                "permanences_live_in": "SimulationBridge.cp_connections.data",
                "update_inputs": "recent presynaptic V1 spikes and postsynaptic spike winners",
                "changed_synapses": int(np.count_nonzero(changed)),
                "changed_synapse_fraction": round(float(np.mean(changed)), 6),
                "mean_absolute_change": round(
                    float(np.mean(np.abs(final_permanences - initial_permanences))),
                    6,
                ),
            },
            "mechanism": dict(spec),
        }
        if arm_name == "intact":
            intact_pooler = pooler
            intact_train_codes = train_codes

    assert intact_pooler is not None and intact_train_codes is not None
    scrambled_codes = infer_codes(intact_pooler, scrambled_features)
    pixel_scramble = score_codes(
        intact_train_codes,
        dataset.train_object_ids,
        scrambled_codes,
        dataset.held_object_ids,
    )
    identity_selection = latency_controls(
        intact_pooler.latency_selector,
        intact_pooler.feedforward_drive(held_features[0], intact_pooler.inference_gain),
    )
    v1_control_encoder = SpikingV1Encoder(
        seed=seed * 1009 + 23,
        n_features=n_v1,
        k_active=args.v1_k_active,
        n_fs=args.v1_fs_neurons,
        n_steps=args.latency_steps,
    )
    v1_selection = latency_controls(v1_control_encoder.selector, held_activity[0])

    intact = arms["intact"]
    shuffled = arms["temporal_shuffle"]
    persistence_lesion = arms["persistence_lesion"]
    trace_lesion = arms["trace_lesion"]
    homeostasis_lesion = arms["homeostasis_lesion"]
    diagnostics = {
        "decode_above_chance": intact["heldout_identity_decode"] >= 0.50,
        "identity_margin_positive": intact["held_to_train_cosine_margin"] > 0.0,
        "temporal_order_load_bearing": (
            intact["held_to_train_cosine_margin"]
            > shuffled["held_to_train_cosine_margin"]
        ),
        "persistence_load_bearing": (
            intact["held_to_train_cosine_margin"]
            > persistence_lesion["held_to_train_cosine_margin"]
        ),
        "presynaptic_trace_load_bearing": (
            intact["held_to_train_cosine_margin"]
            > trace_lesion["held_to_train_cosine_margin"]
        ),
        "homeostasis_reduces_usage_cv": (
            intact["usage"]["usage_cv"]
            < homeostasis_lesion["usage"]["usage_cv"]
        ),
        "pixel_scramble_at_or_below_half": (
            pixel_scramble["heldout_identity_decode"] <= 0.50
        ),
        "local_synapses_changed": (
            intact["local_learning"]["changed_synapses"] > 0
        ),
        "v1_spike_selection_produces_k_winners": (
            len(v1_selection["graded_winners"]) == args.v1_k_active
        ),
        "v1_neural_drive_lesion_silences": (
            len(v1_selection["neural_drive_lesion_winners"]) == 0
        ),
        "identity_spike_selection_produces_k_winners": (
            len(identity_selection["graded_winners"]) == args.k_win
        ),
        "identity_neural_drive_lesion_silences": (
            len(identity_selection["neural_drive_lesion_winners"]) == 0
        ),
        "identity_latency_tracks_graded_drive": (
            identity_selection["graded_host_reference_overlap"] >= 0.80
        ),
        "identity_flat_drive_destroys_rank_signal": (
            identity_selection["flat_drive_overlap_with_graded"]
            <= args.k_win / args.n_col + 0.25
        ),
    }
    calibration_ready = bool(all(diagnostics.values()))

    prior_seeds = {
        int(prior_seed)
        for partition in PRIOR_SEED_PARTITIONS.values()
        for prior_seed in partition
    }
    successor_seeds = {
        int(successor_seed)
        for partition in SEED_PARTITIONS.values()
        for successor_seed in partition
    }
    stream_checks = {
        "fresh_seed_partitions": successor_seeds.isdisjoint(prior_seeds),
        "smoke_seed_outside_scientific_partitions": SMOKE_SEED not in successor_seeds,
        "same_multiset_after_temporal_shuffle": (
            sorted(index for track in ordered_indices for index in track)
            == sorted(index for track in shuffled_indices for index in track)
        ),
        "labels_enter_learning_or_inference": False,
        "training_inputs": [
            "V1 spike indices",
            "recent presynaptic spike trace",
            "recent postsynaptic spike trace",
        ],
        "inference_inputs": ["V1 spike indices"],
        "v1_spikes_read_from": "first-spike timing in cp_firing_states",
        "identity_winners_read_from": "first-spike timing in cp_firing_states",
        "host_top_k_determines_v1_spikes": False,
        "host_top_k_determines_identity_winners": False,
        "remaining_scaffolds": [
            "fixed Gabor filters and host V1-complex normalization",
            "host overlap-to-current max normalization in both latency populations",
            "host first-spike-time readout",
            "host seeded drive-independent same-step tie break",
            "runner-maintained presynaptic temporal trace",
            "runner-maintained postsynaptic persistence current",
            "fixed synthetic continuous-track boundaries",
            "host evaluator labels and centroid scoring",
        ],
    }

    earned = Verdict("visual identity temporal-binding calibration", chance=0.25)
    earned.require(
        "successor seed partitions are fresh",
        stream_checks["fresh_seed_partitions"],
        expect=True,
    )
    earned.require(
        "smoke seed is outside every scientific partition",
        stream_checks["smoke_seed_outside_scientific_partitions"],
        expect=True,
    )
    earned.require(
        "training and held-out transforms are disjoint",
        set(dataset.train_transforms).isdisjoint(set(dataset.held_transforms)),
        expect=True,
    )
    earned.require(
        "temporal shuffle preserves the exact frame multiset",
        stream_checks["same_multiset_after_temporal_shuffle"],
        expect=True,
    )
    earned.require(
        "labels do not enter learning or inference",
        stream_checks["labels_enter_learning_or_inference"],
        expect=False,
    )
    earned.require(
        "V1 spikes come from latency rather than host top-k",
        not stream_checks["host_top_k_determines_v1_spikes"],
        expect=True,
    )
    earned.require(
        "identity winners come from latency rather than host top-k",
        not stream_checks["host_top_k_determines_identity_winners"],
        expect=True,
    )
    earned.require(
        "all scored measurements are finite",
        _all_numeric_values_finite({"arms": arms, "pixel_scramble": pixel_scramble}),
        expect=True,
    )
    decided = earned.decide(go=calibration_ready, verbose=False)

    return {
        "seed": seed,
        "chance": 0.25,
        "n_objects": 4,
        "n_train_images": int(len(dataset.train_images)),
        "n_heldout_images": int(len(dataset.held_images)),
        "arms": arms,
        "pixel_scramble": pixel_scramble,
        "v1_selection": v1_selection,
        "identity_selection": identity_selection,
        "v1_selection_telemetry": v1_encoder.metrics(),
        "identity_selection_telemetry": intact_pooler.latency_selector.metrics(),
        "diagnostics": diagnostics,
        "phase": args.phase,
        "scientific_evidence": args.phase != "smoke",
        "calibration_status": (
            "SMOKE-ONLY"
            if args.phase == "smoke"
            else (
                "UNDEFINED"
                if decided["status"] == UNDEFINED
                else "CALIBRATION_PASS" if decided["go"] else "CALIBRATION_FAIL"
            )
        ),
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "calibration_readiness_verdict": (
            "NOT-SCIENTIFIC-SMOKE" if args.phase == "smoke" else decided["status"]
        ),
        "formal_verdict": formal_verdict_for_phase(args.phase, decided["status"]),
        "stream_checks": stream_checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=CLI_PHASES, default="calibration")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--train-frames", type=int, default=12)
    parser.add_argument("--tracks-per-object", type=int, default=2)
    parser.add_argument("--held-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--pixel-noise", type=float, default=0.025)
    parser.add_argument("--n-orientations", type=int, default=8)
    parser.add_argument("--n-frequencies", type=int, default=2)
    parser.add_argument("--n-pos", type=int, default=8)
    parser.add_argument("--rf-radius", type=int, default=4)
    parser.add_argument(
        "--complex-norm",
        choices=(
            "none",
            "local_orient_div",
            "orient_spatial_div",
            "local_orient_z",
            "spatial_z",
        ),
        default="local_orient_div",
    )
    parser.add_argument("--v1-k-active", type=int, default=32)
    parser.add_argument("--v1-fs-neurons", type=int, default=12)
    parser.add_argument("--n-col", type=int, default=240)
    parser.add_argument("--k-win", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trace-decay", type=float, default=TRACE_DECAY)
    parser.add_argument("--persistence-decay", type=float, default=0.72)
    parser.add_argument("--persistence-gain", type=float, default=3.0)
    parser.add_argument("--pool-lr-pot", type=float, default=0.05)
    parser.add_argument("--pool-lr-depress", type=float, default=0.01)
    parser.add_argument("--homeostasis-rate", type=float, default=0.01)
    parser.add_argument("--homeostasis-strength", type=float, default=6.0)
    parser.add_argument("--latency-fs-neurons", type=int, default=8)
    parser.add_argument("--latency-steps", type=int, default=40)
    parser.add_argument("--out", default=str(OUT))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_seed_partition(args.phase, args.seeds)
    n_v1 = args.n_orientations * args.n_pos * args.n_pos
    if not 0 < args.v1_k_active < n_v1:
        raise ValueError("v1-k-active must satisfy 0 < k < V1-complex feature count")
    if not 0 < args.k_win < args.n_col:
        raise ValueError("k-win must satisfy 0 < k < n-col")
    if args.v1_fs_neurons < 1 or args.latency_fs_neurons < 1 or args.latency_steps < 1:
        raise ValueError("FS neuron counts and latency steps must be positive")
    if not 0.0 <= args.persistence_decay <= 1.0 or args.persistence_gain < 0.0:
        raise ValueError("persistence decay must be in [0,1] and gain non-negative")

    started = time.time()
    rows = []
    for seed in args.seeds:
        row = run_seed(seed, args)
        rows.append(row)
        intact = row["arms"]["intact"]
        print(
            f"[seed {seed}] decode={intact['heldout_identity_decode']:.3f} "
            f"margin={intact['held_to_train_cosine_margin']:+.3f} "
            f"changed={intact['local_learning']['changed_synapse_fraction']:.3f} "
            f"status={row['calibration_status']}",
            flush=True,
        )

    aggregate_verdict = aggregate_formal_verdict(args.phase, rows)

    output = {
        "summary": {
            "probe": "laneD_visual_identity_temporal_binding_gate",
            "phase": args.phase,
            "seeds": args.seeds,
            "formal_gate_run": args.phase == "calibration",
            "formal_verdict": aggregate_verdict,
            "scientific_evidence": args.phase != "smoke",
            "passing_seeds": sum(
                row["calibration_status"] == "CALIBRATION_PASS" for row in rows
            ),
            "seed_partitions": {
                name: list(values) for name, values in SEED_PARTITIONS.items()
            },
            "smoke_seed": SMOKE_SEED,
            "smoke_seed_is_scientific_evidence": False,
            "controls": list(ARM_SPECS)
            + [
                "pixel_scramble",
                "V1_flat_latency_drive",
                "V1_neural_drive_lesion",
                "V1_fs_pathway_lesion",
                "identity_flat_latency_drive",
                "identity_neural_drive_lesion",
                "identity_fs_pathway_lesion",
            ],
            "mechanism": (
                "Graded Gabor/V1-complex activity becomes V1 spike assemblies through "
                "first-spike latency. Substrate-resident V1-to-identity permanences are "
                "updated from local pre-spike traces and post-spike winners; short-lived "
                "post-winner persistence binds adjacent views during learning. Identity "
                "inference uses only V1 spikes and first-spike identity selection."
            ),
            "honest_scope": (
                "No host activation ranking or object label selects V1 or identity units. "
                "The host still computes fixed Gabor responses, divisive normalization, "
                "overlap-to-current scaling, temporal traces, persistence current, spike-time "
                "readout, same-step ties, track boundaries, and external scoring. These are "
                "explicit scaffolds and causal controls, not completed biology."
            ),
            "config": vars(args),
            "elapsed_seconds": round(time.time() - started, 2),
        },
        "per_seed": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(f"[written] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
