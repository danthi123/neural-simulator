"""Bounded calibration gate for stable visual identity from continuous experience.

The retired V2/IT stack is deliberately not used. Images pass through the
existing Gabor/V1-complex encoder into the competitive OnSubstratePooler used by
the successful EMERGE-50 trace experiment. The inherited host top-k winner read
is replaced in both learning and inference by EMERGE-41's spiking rank-order
route: graded column drive becomes current, and the first K columns to spike are
the winners. The candidate mechanism also adds a slow column-usage state that
changes each column's competitive gain, allowing rarely used assemblies to
participate while repeatedly winning assemblies become less excitable.

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
from research.runners._emerge41_fs_wta_kwinners_derisk import (  # noqa: E402
    COL_FS_W as LATENCY_COL_FS_WEIGHT,
    DRIVE_GAIN as LATENCY_DRIVE_GAIN,
    FS_COL_W as LATENCY_FS_COL_WEIGHT,
    N_STEPS as LATENCY_STEPS,
)
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from sim.bridge import SimulationBridge  # noqa: E402
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig  # noqa: E402
from sim.enums import NeuronModel, NeuronType  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402


CALIBRATION_SEEDS = (212, 213)
DEVELOPMENT_SEEDS = (214, 215, 310)
HELDOUT_SEEDS = (311, 312, 313)
SEED_PARTITIONS = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "heldout": HELDOUT_SEEDS,
}
OPEN_PHASES = ("calibration",)

ARM_SPECS = {
    "intact": {"trace": True, "homeostasis": "slow", "shuffle": False, "learning": True},
    "trace_only": {"trace": True, "homeostasis": "epoch", "shuffle": False, "learning": True},
    "temporal_shuffle": {"trace": True, "homeostasis": "slow", "shuffle": True, "learning": True},
    "no_learning": {"trace": False, "homeostasis": "none", "shuffle": False, "learning": False},
    "trace_lesion": {"trace": False, "homeostasis": "slow", "shuffle": False, "learning": True},
    "homeostasis_lesion": {"trace": True, "homeostasis": "none", "shuffle": False, "learning": True},
}

OUT = Path("research/findings/raw/lanes/perception/visual_identity_homeostasis_calibration.json")

LATENCY_DRIVE_RANGE = 6.0
LATENCY_STATE_ARRAYS = (
    "cp_membrane_potential_v",
    "cp_recovery_variable_u",
    "cp_conductance_g_e",
    "cp_conductance_g_i",
    "cp_conductance_g_nmda",
    "cp_conductance_g_nmda_rise",
    "cp_external_input_current",
    "cp_firing_states",
    "cp_prev_firing_states",
    "cp_refractory_timers",
    "cp_neuron_activity_ema",
    "cp_viz_activity_timers",
    "cp_synapse_pulse_timers",
    "cp_synapse_pulse_progress",
)


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
    if phase not in OPEN_PHASES:
        raise ValueError(
            f"phase {phase!r} is not open; this runner is calibration-only and "
            "development/held-out seeds remain locked"
        )
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


class SpikeLatencySelector:
    """Generalized EMERGE-41 first-spike selector with reusable clean windows.

    The host computes the graded dendritic drive from the pooler's on-substrate
    permanences, then only reads spike timing. It never ranks that drive to pick
    winners. Sorting is restricted to first-spike times, with a seeded,
    drive-independent key for same-step ties.
    """

    def __init__(
        self,
        seed: int,
        n_col: int,
        k_win: int,
        n_fs: int,
        n_steps: int,
        wta_enabled: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.n_col = int(n_col)
        self.k_win = int(k_win)
        self.n_fs = int(n_fs)
        self.n_steps = int(n_steps)
        self.wta_enabled = bool(wta_enabled)
        if self.n_col < 2 or not 0 < self.k_win < self.n_col:
            raise ValueError("selector requires 0 < k_win < n_col")
        if self.n_fs < 1 or self.n_steps < 1:
            raise ValueError("selector n_fs and n_steps must be positive")
        self.bridge = self._build_bridge()
        self.column_indices = np.asarray(self.bridge.region_manager.indices("latency_columns"), dtype=np.int64)
        self._tie_rng = np.random.default_rng(self.seed * 101 + 17)
        self._initial_state = {
            name: getattr(self.bridge, name).copy()
            for name in LATENCY_STATE_ARRAYS
            if getattr(self.bridge, name, None) is not None
        }
        self.selection_calls = 0
        self.total_fired = 0
        self.last_first_spike = np.full(self.n_col, self.n_steps + 1, dtype=np.int64)
        self.last_fired_count = 0

    def _build_bridge(self) -> SimulationBridge:
        excitatory = dict(
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        )
        regions = [BrainRegion(name="latency_columns", n_neurons=self.n_col, **excitatory)]
        pathways: list[RegionPathway] = []
        if self.wta_enabled:
            regions.append(
                BrainRegion(
                    name="latency_fs",
                    n_neurons=self.n_fs,
                    exc_fraction=0.0,
                    internal_density=0.0,
                    exc_weight_mean=0.0,
                    inh_weight_mean=0.0,
                    weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                )
            )
            pathways.extend(
                (
                    RegionPathway(
                        from_region="latency_columns",
                        to_region="latency_fs",
                        density=1.0,
                        weight_mean=LATENCY_COL_FS_WEIGHT,
                        weight_jitter=0.0,
                        plastic=False,
                    ),
                    RegionPathway(
                        from_region="latency_fs",
                        to_region="latency_columns",
                        density=1.0,
                        weight_mean=LATENCY_FS_COL_WEIGHT,
                        weight_jitter=0.0,
                        plastic=False,
                    ),
                )
            )

        config = CoreSimConfig()
        config.seed = config.heterogeneity_seed = config.ou_seed = self.seed
        config.dt_ms = 1.0
        config.num_traits = 1
        config.neuron_model_type = NeuronModel.IZHIKEVICH.name
        config.neural_profile_name = "GENERIC_UNSTRUCTURED"
        config.connections_per_neuron = 0
        config.enable_brain_region_framework = True
        config.brain_regions = regions
        config.region_pathways = pathways
        config.enable_stdp = False
        config.enable_hebbian_learning = False
        config.enable_nmda = False
        config.fast_spike_reset = True
        for field in (
            "enable_homeostasis",
            "enable_short_term_plasticity",
            "enable_ou_process",
            "enable_conductance_noise",
            "enable_parameter_heterogeneity",
            "enable_structural_plasticity",
        ):
            setattr(config, field, False)
        config.enable_coincidence_detection = False
        bridge = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=GPUConfig(),
        )
        bridge.runtime_state.max_delay_steps = int(config.max_synaptic_delay_ms / config.dt_ms)
        bridge.runtime_state.actual_seed_used = self.seed
        bridge._initialize_simulation_data(called_from_playback_init=False)
        return bridge

    def _reset_window(self) -> None:
        for name, initial in self._initial_state.items():
            getattr(self.bridge, name)[:] = initial
        self.bridge.runtime_state.current_time_ms = 0.0
        self.bridge.runtime_state.current_time_step = 0

    def select(self, drive: np.ndarray, neural_drive_enabled: bool = True) -> set[int]:
        values = np.asarray(drive, dtype=np.float64)
        if values.shape != (self.n_col,):
            raise ValueError(f"drive must have shape ({self.n_col},), got {values.shape}")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("drive must be finite and non-negative")

        self._reset_window()
        peak = float(values.max()) if values.size else 0.0
        scaled = values * (LATENCY_DRIVE_RANGE / peak) if peak > 0.0 else values
        current = np.zeros(self.bridge.cp_external_input_current.shape[0], dtype=np.float32)
        if neural_drive_enabled:
            current[self.column_indices] = (LATENCY_DRIVE_GAIN * scaled).astype(np.float32)
        self.bridge.cp_external_input_current[:] = current

        first_spike = np.full(self.n_col, self.n_steps + 1, dtype=np.int64)
        for step in range(self.n_steps):
            self.bridge._run_one_simulation_step()
            fired_now = np.asarray(self.bridge.cp_firing_states)[self.column_indices].astype(bool)
            newly = fired_now & (first_spike > self.n_steps)
            first_spike[newly] = step
        self.bridge.cp_external_input_current[:] = 0.0

        fired = np.flatnonzero(first_spike <= self.n_steps)
        self.selection_calls += 1
        self.last_first_spike = first_spike
        self.last_fired_count = int(fired.size)
        self.total_fired += int(fired.size)
        if fired.size == 0:
            return set()
        tie_break = self._tie_rng.random(fired.size)
        order = fired[np.lexsort((tie_break, first_spike[fired]))]
        return set(int(column) for column in order[: self.k_win])

    def metrics(self) -> dict[str, float | int | bool | str]:
        mean_fired = self.total_fired / max(self.selection_calls, 1)
        return {
            "selection_route": "first_spike_latency",
            "winner_source": "cp_firing_states timing",
            "host_drive_ranking_used_for_winners": False,
            "host_spike_time_readout": True,
            "host_same_step_tie_break": "seeded drive-independent key",
            "fs_pathways_enabled": self.wta_enabled,
            "selection_calls": self.selection_calls,
            "mean_columns_fired_per_window": round(float(mean_fired), 4),
            "last_columns_fired": self.last_fired_count,
        }


class HomeostaticTracePooler(TraceV1Pooler):
    """Trace pooler with slow, online column-usage homeostasis."""

    def __init__(
        self,
        *args,
        homeostasis_rate: float,
        homeostasis_strength: float,
        latency_fs_neurons: int,
        latency_steps: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.homeostasis_rate = float(homeostasis_rate)
        self.homeostasis_strength = float(homeostasis_strength)
        self.target_usage = self.k_win / self.n_col
        self.usage = np.full(self.n_col, self.target_usage, dtype=np.float64)
        self.selection_counts = np.zeros(self.n_col, dtype=np.int64)
        self.inference_gain = np.ones(self.n_col, dtype=np.float64)
        self.latency_selector = SpikeLatencySelector(
            seed=int(kwargs["seed"]) * 17 + 5,
            n_col=self.n_col,
            k_win=self.k_win,
            n_fs=latency_fs_neurons,
            n_steps=latency_steps,
        )

    def _winners(self, features: set[int], boost: np.ndarray | None = None) -> set[int]:
        """Select pooler columns only from the latency selector's spike times."""
        return self.latency_selector.select(self._drive(features, boost))

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
        latency_fs_neurons=args.latency_fs_neurons,
        latency_steps=args.latency_steps,
    )


def selection_controls(
    pooler: HomeostaticTracePooler,
    features: set[int],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float | int | bool | list[int]]:
    """Lesion the neural input and flatten its ranking signal on one fixed drive."""
    drive = pooler._drive(features, pooler.inference_gain)
    selector = pooler.latency_selector
    graded_winners = selector.select(drive)
    graded_fired_count = selector.last_fired_count
    host_reference = set(int(column) for column in np.argsort(-drive)[: pooler.k_win])
    flat_winners = selector.select(np.full(pooler.n_col, float(np.mean(drive))))
    silent_winners = selector.select(drive, neural_drive_enabled=False)

    fs_lesion = SpikeLatencySelector(
        seed=seed,
        n_col=pooler.n_col,
        k_win=pooler.k_win,
        n_fs=args.latency_fs_neurons,
        n_steps=args.latency_steps,
        wta_enabled=False,
    )
    lesion_winners = fs_lesion.select(drive)
    denominator = max(pooler.k_win, 1)
    return {
        "graded_winners": sorted(graded_winners),
        "graded_host_reference_overlap": round(len(graded_winners & host_reference) / denominator, 4),
        "flat_drive_overlap_with_graded": round(len(flat_winners & graded_winners) / denominator, 4),
        "flat_drive_winners": sorted(flat_winners),
        "neural_drive_lesion_winners": sorted(silent_winners),
        "fs_lesion_winner_overlap": round(len(lesion_winners & graded_winners) / denominator, 4),
        "fs_intact_columns_fired": graded_fired_count,
        "fs_lesion_columns_fired": fs_lesion.last_fired_count,
        "pooler_host_drive_ranking_used_for_winners": False,
    }


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
    neural_selection = selection_controls(
        intact_pooler,
        held_features[0],
        seed=pool_seed * 17 + 5,
        args=args,
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
        "neural_selection_produces_k_winners": (
            len(neural_selection["graded_winners"]) == args.k_win
        ),
        "neural_drive_lesion_silences_selection": (
            len(neural_selection["neural_drive_lesion_winners"]) == 0
        ),
        "latency_tracks_graded_drive": (
            neural_selection["graded_host_reference_overlap"] >= 0.80
        ),
        "flat_drive_destroys_rank_signal": (
            neural_selection["flat_drive_overlap_with_graded"]
            <= args.k_win / args.n_col + 0.25
        ),
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
        "neural_selection": neural_selection,
        "selection_telemetry": intact_pooler.latency_selector.metrics(),
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
            "pooler_winners_read_from": "first-spike timing in cp_firing_states",
            "host_top_k_determines_pooler_winners": False,
            "remaining_selection_scaffolds": [
                "host overlap-to-current max normalization",
                "host first-spike-time readout",
                "host seeded drive-independent same-step tie break",
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=OPEN_PHASES, default="calibration")
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
    parser.add_argument("--latency-fs-neurons", type=int, default=8)
    parser.add_argument("--latency-steps", type=int, default=LATENCY_STEPS)
    parser.add_argument("--out", default=str(OUT))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_seed_partition(args.phase, args.seeds)
    if args.t_active > args.n_orientations * args.n_pos * args.n_pos:
        raise ValueError("t_active exceeds the V1-complex feature count")
    if args.k_win >= args.n_col:
        raise ValueError("k_win must be smaller than n_col")
    if args.latency_fs_neurons < 1 or args.latency_steps < 1:
        raise ValueError("latency-fs-neurons and latency-steps must be positive")
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
            "controls": list(ARM_SPECS) + [
                "pixel_scramble",
                "flat_latency_drive",
                "neural_drive_lesion",
                "fs_pathway_lesion",
            ],
            "mechanism": (
                "Gabor/V1-complex activity drives competitive OnSubstratePooler assemblies. "
                "Foldiak traces update cp_connections through committed learning kernels; graded "
                "column drive is converted to current and first-spike latency selects every training "
                "and inference winner; an online EMA of each column's wins changes its intrinsic gain."
            ),
            "honest_scope": (
                "Pooler permanences, learning, and winner selection use the simulation substrate. The "
                "host still computes dendritic overlap from substrate permanences, max-normalizes it into "
                "the calibrated current range, reads first-spike times, and breaks same-step ties with a "
                "seeded drive-independent key. No host ranking of the graded drive chooses winners. FS "
                "inhibition sparsifies later firing, while EMERGE-41 established that rank-order latency "
                "selects the winners. V1 feature sparsification and evaluator labels remain host-side; "
                "labels never enter learning or inference."
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
