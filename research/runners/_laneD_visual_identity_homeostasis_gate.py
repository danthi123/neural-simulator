"""Bounded calibration gate for stable visual identity from continuous experience.

The retired V2/IT stack is deliberately not used. Images pass through the
existing Gabor/V1-complex encoder into the competitive OnSubstratePooler used by
the successful EMERGE-50 trace experiment. The candidate mechanism adds a slow
column-usage state that changes each column's competitive gain, allowing rarely
used assemblies to participate while repeatedly winning assemblies become less
excitable.

Four synthetic objects are viewed along continuous position, scale, and modest
lighting trajectories. Object identity is never passed to the pooler or to
``infer_codes``. Labels remain outside the network and are used only after
inference to score its spike-selected assembly codes.

Calibration arms:
  * intact: trace learning plus slow online usage homeostasis;
  * trace_only: the prior trace learner with epoch-level boosting;
  * temporal_shuffle: the intact mechanism given the same frames out of order;
  * no_learning: initial random permanences, never updated;
  * trace_lesion: current input only, with slow homeostasis intact;
  * homeostasis_lesion: trace learning with fixed column gain;
  * pixel_scramble: intact network, held-out images spatially scrambled.

This first version is calibration-only evidence. It reports provisional
diagnostics and never emits a formal GO. Seeds 212/213 are calibration seeds;
214/215/310 are reserved for development and 311/312/313 for a later held-out
gate.

Cheap smoke:
  SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -u -m \
    research.runners._laneD_visual_identity_homeostasis_gate \
    --phase calibration --seeds 212 --epochs 1 --train-frames 4 \
    --tracks-per-object 1 --held-frames 2 --image-size 16 --n-pos 4 \
    --n-col 24 --k-win 3 --t-active 8 --out /tmp/visual_identity_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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

from research.runners._laneD_v1_pooler_trace_invariance_derisk import (  # noqa: E402
    TraceV1Pooler,
    _binary_codes,
    _centroid_decode,
    _held_train_margin,
    _normalize_complex,
    _scramble_images,
    _top_features,
)
from research.runners._emerge50_trace_rule_derisk import (  # noqa: E402
    TRACE_DECAY,
    _apply_traced_potentiation,
)
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)


CALIBRATION_SEEDS = (212, 213)
DEVELOPMENT_SEEDS = (214, 215, 310)
HELDOUT_SEEDS = (311, 312, 313)
SEED_PARTITIONS = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "heldout": HELDOUT_SEEDS,
}

ARM_SPECS = {
    "intact": {"trace": True, "homeostasis": "slow", "shuffle": False, "learning": True},
    "trace_only": {"trace": True, "homeostasis": "epoch", "shuffle": False, "learning": True},
    "temporal_shuffle": {"trace": True, "homeostasis": "slow", "shuffle": True, "learning": True},
    "no_learning": {"trace": False, "homeostasis": "none", "shuffle": False, "learning": False},
    "trace_lesion": {"trace": False, "homeostasis": "slow", "shuffle": False, "learning": True},
    "homeostasis_lesion": {"trace": True, "homeostasis": "none", "shuffle": False, "learning": True},
}

OUT = Path("research/findings/raw/lanes/perception/visual_identity_homeostasis_calibration.json")


@dataclass(frozen=True)
class Transform:
    x: float
    y: float
    scale: float
    lighting: float


@dataclass
class VisualDataset:
    train_images: np.ndarray
    train_object_ids: np.ndarray
    train_transforms: tuple[Transform, ...]
    train_tracks: tuple[tuple[int, ...], ...]
    held_images: np.ndarray
    held_object_ids: np.ndarray
    held_transforms: tuple[Transform, ...]


# Line segments in object-local coordinates. Identity is the conjunction and
# arrangement of edges; position, scale, and lighting are nuisance variables.
OBJECT_SEGMENTS: tuple[tuple[tuple[float, float, float, float], ...], ...] = (
    ((-0.65, -0.70, -0.65, 0.70), (-0.65, 0.70, 0.65, 0.70)),  # L
    ((-0.70, -0.60, 0.70, -0.60), (0.00, -0.60, 0.00, 0.72)),  # T
    ((-0.65, -0.65, 0.65, 0.65), (-0.65, 0.65, 0.65, -0.65)),  # X
    ((-0.62, -0.68, -0.62, 0.55), (0.62, -0.68, 0.62, 0.55),
     (-0.62, 0.55, 0.62, 0.55)),  # U
)


def validate_seed_partition(phase: str, seeds: Sequence[int]) -> None:
    """Prevent calibration from consuming seeds reserved for later decisions."""
    allowed = set(SEED_PARTITIONS[phase])
    unexpected = sorted(set(int(seed) for seed in seeds) - allowed)
    if unexpected:
        raise ValueError(f"{phase} phase cannot use seeds {unexpected}; allowed seeds are {sorted(allowed)}")


def _segment_response(
    xx: np.ndarray,
    yy: np.ndarray,
    segment: tuple[float, float, float, float],
    transform: Transform,
    image_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = segment
    radius = image_size * 0.235 * transform.scale
    cx = image_size * (0.5 + transform.x)
    cy = image_size * (0.5 + transform.y)
    ax, ay = cx + x0 * radius, cy + y0 * radius
    bx, by = cx + x1 * radius, cy + y1 * radius
    vx, vy = bx - ax, by - ay
    denom = max(vx * vx + vy * vy, 1e-6)
    projection = np.clip(((xx - ax) * vx + (yy - ay) * vy) / denom, 0.0, 1.0)
    dx = xx - (ax + projection * vx)
    dy = yy - (ay + projection * vy)
    sigma = max(0.65, image_size * 0.035 * transform.scale)
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32)


def render_object(
    object_id: int,
    transform: Transform,
    image_size: int,
    rng: np.random.Generator,
    pixel_noise: float,
) -> np.ndarray:
    """Render one object into the ON/OFF layout consumed by the V1 encoder."""
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    on = np.zeros((image_size, image_size), dtype=np.float32)
    for segment in OBJECT_SEGMENTS[object_id]:
        on = np.maximum(on, _segment_response(xx, yy, segment, transform, image_size))
    on = np.clip(on * transform.lighting, 0.0, 1.0)
    gx = np.gradient(on, axis=1)
    gy = np.gradient(on, axis=0)
    off = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    off = off / (float(off.max()) + 1e-6) * min(0.45, transform.lighting * 0.45)
    on = np.clip(on + rng.normal(0.0, pixel_noise, on.shape).astype(np.float32), 0.0, 1.0)
    off = np.clip(off + rng.normal(0.0, pixel_noise * 0.5, off.shape).astype(np.float32), 0.0, 1.0)
    return np.stack((on, off), axis=0)


def _continuous_track(n_frames: int, track_index: int) -> tuple[Transform, ...]:
    """A smooth camera/object trajectory; adjacent frames vary only slightly."""
    phase = track_index * 0.73
    ts = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    return tuple(
        Transform(
            x=float(0.13 * math.sin(2.0 * math.pi * t + phase)),
            y=float(0.09 * math.cos(2.0 * math.pi * t + phase * 0.7)),
            scale=float(0.90 + 0.14 * (0.5 + 0.5 * math.sin(2.0 * math.pi * t + 0.4 + phase))),
            lighting=float(0.72 + 0.23 * (0.5 + 0.5 * math.cos(2.0 * math.pi * t + 0.8 + phase))),
        )
        for t in ts
    )


def _held_transforms(n_frames: int, object_id: int) -> tuple[Transform, ...]:
    """Transformation combinations absent from the continuous training tracks."""
    templates = (
        Transform(-0.18, -0.13, 0.76, 0.62),
        Transform(0.18, 0.13, 1.16, 1.00),
        Transform(-0.18, 0.13, 1.13, 0.66),
        Transform(0.18, -0.13, 0.79, 0.98),
        Transform(-0.16, 0.00, 1.18, 0.78),
        Transform(0.16, 0.00, 0.74, 0.84),
        Transform(0.00, -0.14, 1.15, 0.64),
        Transform(0.00, 0.14, 0.77, 1.00),
    )
    offset = object_id % len(templates)
    return tuple(templates[(offset + i) % len(templates)] for i in range(n_frames))


def build_visual_dataset(
    seed: int,
    image_size: int = 32,
    train_frames: int = 12,
    tracks_per_object: int = 2,
    held_frames: int = 8,
    pixel_noise: float = 0.025,
) -> VisualDataset:
    """Build unlabeled experience tracks plus labels held separately for scoring."""
    if train_frames < 2 or held_frames < 1 or tracks_per_object < 1:
        raise ValueError("train_frames >= 2, held_frames >= 1, and tracks_per_object >= 1 are required")
    rng = np.random.default_rng(seed * 1009 + 17)
    train_images: list[np.ndarray] = []
    train_ids: list[int] = []
    train_transforms: list[Transform] = []
    train_tracks: list[tuple[int, ...]] = []
    held_images: list[np.ndarray] = []
    held_ids: list[int] = []
    held_transforms: list[Transform] = []

    for object_id in range(len(OBJECT_SEGMENTS)):
        for track_number in range(tracks_per_object):
            transforms = _continuous_track(train_frames, object_id * tracks_per_object + track_number)
            track_indices = []
            for transform in transforms:
                track_indices.append(len(train_images))
                train_images.append(render_object(object_id, transform, image_size, rng, pixel_noise))
                train_ids.append(object_id)
                train_transforms.append(transform)
            train_tracks.append(tuple(track_indices))
        for transform in _held_transforms(held_frames, object_id):
            held_images.append(render_object(object_id, transform, image_size, rng, pixel_noise))
            held_ids.append(object_id)
            held_transforms.append(transform)

    return VisualDataset(
        train_images=np.asarray(train_images, dtype=np.float32),
        train_object_ids=np.asarray(train_ids, dtype=np.int64),
        train_transforms=tuple(train_transforms),
        train_tracks=tuple(train_tracks),
        held_images=np.asarray(held_images, dtype=np.float32),
        held_object_ids=np.asarray(held_ids, dtype=np.int64),
        held_transforms=tuple(held_transforms),
    )


def shuffled_track_indices(tracks: Sequence[Sequence[int]], seed: int) -> tuple[tuple[int, ...], ...]:
    """Destroy temporal continuity while preserving the exact frame multiset."""
    lengths = [len(track) for track in tracks]
    flat = np.asarray([index for track in tracks for index in track], dtype=np.int64)
    np.random.default_rng(seed).shuffle(flat)
    out = []
    cursor = 0
    for length in lengths:
        out.append(tuple(int(index) for index in flat[cursor: cursor + length]))
        cursor += length
    return tuple(out)


def update_usage_ema(usage: np.ndarray, winners: Iterable[int], rate: float) -> np.ndarray:
    """Local slow estimate of how often each assembly wins."""
    updated = np.asarray(usage, dtype=np.float64) * (1.0 - rate)
    winner_array = np.fromiter((int(winner) for winner in winners), dtype=np.int64)
    if winner_array.size:
        updated[winner_array] += rate
    return updated


def homeostatic_gain(usage: np.ndarray, target: float, strength: float) -> np.ndarray:
    """Usage-dependent intrinsic gain, bounded to avoid unstable competition."""
    return np.clip(np.exp(strength * (target - np.asarray(usage))), 0.20, 5.0)


class HomeostaticTracePooler(TraceV1Pooler):
    """Trace pooler with slow, online column-usage homeostasis."""

    def __init__(self, *args, homeostasis_rate: float, homeostasis_strength: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.homeostasis_rate = float(homeostasis_rate)
        self.homeostasis_strength = float(homeostasis_strength)
        self.target_usage = self.k_win / self.n_col
        self.usage = np.full(self.n_col, self.target_usage, dtype=np.float64)
        self.selection_counts = np.zeros(self.n_col, dtype=np.int64)
        self.inference_gain = np.ones(self.n_col, dtype=np.float64)

    def train_tracks(
        self,
        tracks: Sequence[Sequence[set[int]]],
        epochs: int,
        trace_decay: float,
        trace_enabled: bool,
        homeostasis_mode: str,
        seed: int,
    ) -> None:
        if homeostasis_mode not in {"slow", "epoch", "none"}:
            raise ValueError(f"unknown homeostasis mode: {homeostasis_mode}")
        rng = np.random.default_rng(seed)
        epoch_gain = np.ones(self.n_col, dtype=np.float64)
        cumulative = np.zeros(self.n_col, dtype=np.float64)
        presentations = 0
        order = np.arange(len(tracks))
        for _ in range(epochs):
            rng.shuffle(order)
            for track_index in order:
                trace = np.zeros(self.n_in, dtype=np.float64)
                for features in tracks[int(track_index)]:
                    active = np.zeros(self.n_in, dtype=np.float64)
                    active[list(features)] = 1.0
                    trace = active if not trace_enabled else np.clip(trace * trace_decay + active, 0.0, 1.0)
                    if homeostasis_mode == "slow":
                        gain = homeostatic_gain(self.usage, self.target_usage, self.homeostasis_strength)
                    elif homeostasis_mode == "epoch":
                        gain = epoch_gain
                    else:
                        gain = None
                    winners = self._winners(features, gain)
                    _apply_traced_potentiation(self, trace, winners, self.lp)
                    self._winner_inactive_traced(winners, trace, self.ld_wi)
                    winner_array = np.fromiter((int(winner) for winner in winners), dtype=np.int64)
                    self.selection_counts[winner_array] += 1
                    cumulative[winner_array] += 1.0
                    presentations += 1
                    self.usage = update_usage_ema(self.usage, winners, self.homeostasis_rate)
            if homeostasis_mode == "epoch":
                observed = cumulative / max(presentations, 1)
                epoch_gain = homeostatic_gain(observed, self.target_usage, 2.0)

        if homeostasis_mode == "slow":
            self.inference_gain = homeostatic_gain(self.usage, self.target_usage, self.homeostasis_strength)
        elif homeostasis_mode == "epoch":
            self.inference_gain = epoch_gain

    def codon(self, features: set[int]) -> set[int]:
        return self._winners(features, self.inference_gain)

    def usage_metrics(self) -> dict[str, float]:
        counts = self.selection_counts.astype(np.float64)
        mean = float(counts.mean())
        active = int(np.count_nonzero(counts))
        probabilities = counts / max(float(counts.sum()), 1.0)
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
        normalized_entropy = entropy / max(math.log(self.n_col), 1e-9)
        return {
            "active_columns": active,
            "dead_column_fraction": round(1.0 - active / self.n_col, 4),
            "usage_cv": round(float(counts.std()) / max(mean, 1e-9), 4),
            "normalized_usage_entropy": round(normalized_entropy, 4),
            "max_to_mean_usage": round(float(counts.max()) / max(mean, 1e-9), 4),
        }


def encode_images(images: np.ndarray, args: argparse.Namespace, response_matrix: np.ndarray) -> list[set[int]]:
    """Pixels to sparse V1-complex activity; no identity information enters."""
    v1 = pool_v1_to_complex(
        encode_v1(images, response_matrix), args.n_orientations, args.n_frequencies, args.n_pos
    )
    v1 = _normalize_complex(v1, args.complex_norm, args.n_orientations, args.n_pos)
    return _top_features(v1, args.t_active)


def infer_codes(pooler: HomeostaticTracePooler, features: Sequence[set[int]]) -> np.ndarray:
    """Infer assembly codes from V1 activity alone; labels are intentionally absent."""
    return _binary_codes([pooler.codon(feature_set) for feature_set in features], pooler.n_col)


def _jaccard_stability(codes: np.ndarray, object_ids: np.ndarray) -> tuple[float, float, float]:
    within: list[float] = []
    cross: list[float] = []
    active = codes > 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            union = int(np.logical_or(active[i], active[j]).sum())
            similarity = float(np.logical_and(active[i], active[j]).sum() / max(union, 1))
            (within if object_ids[i] == object_ids[j] else cross).append(similarity)
    within_mean = float(np.mean(within)) if within else 0.0
    cross_mean = float(np.mean(cross)) if cross else 0.0
    return within_mean, cross_mean, within_mean - cross_mean


def score_codes(
    train_codes: np.ndarray,
    train_ids: np.ndarray,
    held_codes: np.ndarray,
    held_ids: np.ndarray,
) -> dict[str, float]:
    same_cos, cross_cos, cosine_margin = _held_train_margin(train_codes, train_ids, held_codes, held_ids)
    within_j, cross_j, jaccard_margin = _jaccard_stability(held_codes, held_ids)
    return {
        "heldout_identity_decode": round(_centroid_decode(train_codes, train_ids, held_codes, held_ids), 4),
        "held_to_train_same_cosine": round(same_cos, 4),
        "held_to_train_cross_cosine": round(cross_cos, 4),
        "held_to_train_cosine_margin": round(cosine_margin, 4),
        "held_within_object_jaccard": round(within_j, 4),
        "held_cross_object_jaccard": round(cross_j, 4),
        "held_jaccard_margin": round(jaccard_margin, 4),
        "mean_code_activity": round(float(held_codes.mean()), 4),
    }


def _make_pooler(seed: int, n_in: int, args: argparse.Namespace) -> HomeostaticTracePooler:
    return HomeostaticTracePooler(
        seed=seed,
        n_in=n_in,
        n_col=args.n_col,
        k_win=args.k_win,
        lp=args.pool_lr_pot,
        ld_wi=args.pool_lr_depress,
        homeostasis_rate=args.homeostasis_rate,
        homeostasis_strength=args.homeostasis_strength,
    )


def run_seed(seed: int, args: argparse.Namespace) -> dict:
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
    train_features = encode_images(dataset.train_images, args, response_matrix)
    held_features = encode_images(dataset.held_images, args, response_matrix)
    scrambled_images = _scramble_images(dataset.held_images, seed * 101 + 83)
    scrambled_features = encode_images(scrambled_images, args, response_matrix)
    ordered_indices = dataset.train_tracks
    shuffled_indices = shuffled_track_indices(ordered_indices, seed * 101 + 89)
    pool_seed = seed * 1009 + 31
    arms: dict[str, dict] = {}

    intact_pooler: HomeostaticTracePooler | None = None
    intact_train_codes: np.ndarray | None = None
    for arm_name, spec in ARM_SPECS.items():
        n_v1_complex = args.n_orientations * args.n_pos * args.n_pos
        pooler = _make_pooler(pool_seed, n_v1_complex, args)
        # The V1-complex feature space excludes frequency after phase/frequency pooling.
        assert pooler.n_in == args.n_orientations * args.n_pos * args.n_pos
        indices = shuffled_indices if spec["shuffle"] else ordered_indices
        tracks = tuple(tuple(train_features[index] for index in track) for track in indices)
        if spec["learning"]:
            pooler.train_tracks(
                tracks=tracks,
                epochs=args.epochs,
                trace_decay=args.trace_decay,
                trace_enabled=bool(spec["trace"]),
                homeostasis_mode=str(spec["homeostasis"]),
                seed=seed * 1009 + 43,
            )
        train_codes = infer_codes(pooler, train_features)
        held_codes = infer_codes(pooler, held_features)
        arms[arm_name] = {
            **score_codes(train_codes, dataset.train_object_ids, held_codes, dataset.held_object_ids),
            "usage": pooler.usage_metrics(),
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

    intact = arms["intact"]
    shuffled = arms["temporal_shuffle"]
    trace_lesion = arms["trace_lesion"]
    homeostasis_lesion = arms["homeostasis_lesion"]
    diagnostics = {
        "decode_above_chance": intact["heldout_identity_decode"] >= 0.50,
        "identity_margin_positive": intact["held_to_train_cosine_margin"] > 0.0,
        "trace_directional_vs_shuffle": (
            intact["held_to_train_cosine_margin"] > shuffled["held_to_train_cosine_margin"]
        ),
        "trace_directional_vs_lesion": (
            intact["held_to_train_cosine_margin"] > trace_lesion["held_to_train_cosine_margin"]
        ),
        "homeostasis_reduces_usage_cv": (
            intact["usage"]["usage_cv"] < homeostasis_lesion["usage"]["usage_cv"]
        ),
        "pixel_scramble_at_or_below_half": pixel_scramble["heldout_identity_decode"] <= 0.50,
    }
    calibration_ready = bool(all(diagnostics.values()))
    return {
        "seed": seed,
        "chance": 0.25,
        "n_objects": len(OBJECT_SEGMENTS),
        "n_train_images": int(len(dataset.train_images)),
        "n_heldout_images": int(len(dataset.held_images)),
        "arms": arms,
        "pixel_scramble": pixel_scramble,
        "diagnostics": diagnostics,
        "calibration_status": "CANDIDATE" if calibration_ready else "NEEDS-REVISION",
        "formal_verdict": "NOT-RUN-CALIBRATION-ONLY",
        "stream_checks": {
            "same_multiset_after_temporal_shuffle": (
                sorted(index for track in ordered_indices for index in track)
                == sorted(index for track in shuffled_indices for index in track)
            ),
            "labels_enter_training_or_inference": False,
            "inference_inputs": ["V1-complex active feature indices"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(SEED_PARTITIONS), default="calibration")
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
        choices=("none", "local_orient_div", "orient_spatial_div", "local_orient_z", "spatial_z"),
        default="local_orient_div",
    )
    parser.add_argument("--t-active", type=int, default=32)
    parser.add_argument("--n-col", type=int, default=240)
    parser.add_argument("--k-win", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trace-decay", type=float, default=TRACE_DECAY)
    parser.add_argument("--pool-lr-pot", type=float, default=0.05)
    parser.add_argument("--pool-lr-depress", type=float, default=0.01)
    parser.add_argument("--homeostasis-rate", type=float, default=0.01)
    parser.add_argument("--homeostasis-strength", type=float, default=6.0)
    parser.add_argument("--out", default=str(OUT))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_seed_partition(args.phase, args.seeds)
    if args.t_active > args.n_orientations * args.n_pos * args.n_pos:
        raise ValueError("t_active exceeds the V1-complex feature count")
    if args.k_win >= args.n_col:
        raise ValueError("k_win must be smaller than n_col")
    started = time.time()
    rows = []
    for seed in args.seeds:
        row = run_seed(seed, args)
        rows.append(row)
        intact = row["arms"]["intact"]
        print(
            f"[seed {seed}] decode={intact['heldout_identity_decode']:.3f} "
            f"margin={intact['held_to_train_cosine_margin']:+.3f} "
            f"usage_cv={intact['usage']['usage_cv']:.3f} "
            f"status={row['calibration_status']}",
            flush=True,
        )

    output = {
        "summary": {
            "probe": "laneD_visual_identity_homeostasis_gate",
            "phase": args.phase,
            "seeds": args.seeds,
            "formal_gate_run": False,
            "formal_verdict": "NOT-RUN-CALIBRATION-ONLY",
            "candidate_seeds": sum(row["calibration_status"] == "CANDIDATE" for row in rows),
            "seed_partitions": {name: list(values) for name, values in SEED_PARTITIONS.items()},
            "controls": list(ARM_SPECS) + ["pixel_scramble"],
            "mechanism": (
                "Gabor/V1-complex activity drives competitive OnSubstratePooler assemblies. "
                "Foldiak traces update cp_connections through committed learning kernels; an online "
                "EMA of each column's wins changes its intrinsic competitive gain."
            ),
            "honest_scope": (
                "Pooler permanences and learning use the shared simulation substrate. Winner selection "
                "retains the inherited host top-k calibration scaffold; labels are evaluator-only and "
                "never enter training or inference."
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
