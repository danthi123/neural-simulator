"""Smoke-first visual hierarchy successor: local V2 parts to trace-bound IT identity.

Fixed Gabor/V1-complex responses are converted to V1 spikes by the established
latency encoder. A retinotopic V2 layer then learns local orientation
combinations through substrate-resident permanences and per-hypercolumn
excitatory/FS competition. Every V2 excitatory cell that fires by the fixed
deadline is forwarded; no host top-k or first-K truncation defines the V2 code.
An IT population learns from those V2 spike sets with a presynaptic Foldiak
trace. The predecessor's postsynaptic persistence current is absent.

Seed 222 is non-scientific smoke only. Calibration requires the exact ordered
tuple 503/509. Development and held-out seeds are locked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._emerge14_stageC_onbridge_learning_derisk import _host  # noqa: E402
from research.runners._emerge46_spiking_stacked_pooler_derisk import _build_cells_bridge  # noqa: E402
from research.runners._emerge50_trace_rule_derisk import (  # noqa: E402
    TRACE_DECAY,
    _apply_traced_potentiation,
)
from research.runners._laneD_v1_pooler_trace_invariance_derisk import (  # noqa: E402
    TraceV1Pooler,
    _binary_codes,
    _scramble_images,
)
from research.runners._laneD_visual_identity_homeostasis_gate import (  # noqa: E402
    SpikeLatencySelector,
    build_visual_dataset,
    score_codes,
    shuffled_track_indices,
)
from research.runners._laneD_visual_identity_temporal_binding_gate import (  # noqa: E402
    SpikingV1Encoder,
    dense_v1_activity,
)
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
)
from sim.kernels import (  # noqa: E402
    fused_htm_permanence_update,
    fused_htm_winner_inactive_depression,
)
from tools.verdict import GO, NO_GO, UNDEFINED, Verdict  # noqa: E402


SMOKE_SEED = 222
CALIBRATION_SEEDS = (503, 509)
DEVELOPMENT_SEEDS = (521, 523, 541)
HELDOUT_SEEDS = (547, 557, 563)
SEED_PARTITIONS = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "heldout": HELDOUT_SEEDS,
}
OPEN_PHASES = ("calibration",)
CLI_PHASES = ("smoke",) + OPEN_PHASES

ARM_SPECS = {
    "intact": dict(v2_learning=True, it_learning=True, trace=True),
    "all_learning_off": dict(v2_learning=False, it_learning=False, trace=False),
    "v2_learning_off": dict(v2_learning=False, it_learning=True, trace=True),
    "it_trace_off": dict(v2_learning=True, it_learning=True, trace=False),
    "temporal_shuffle": dict(
        v2_learning=True, it_learning=True, trace=True, temporal_shuffle=True
    ),
    "v2_local_fs_lesion": dict(
        v2_learning=True, it_learning=True, trace=True, v2_fs=False
    ),
    "it_fs_lesion": dict(
        v2_learning=True, it_learning=True, trace=True, it_fs=False
    ),
    "receptive_field_scramble": dict(
        v2_learning=True, it_learning=True, trace=True, rf_scramble=True
    ),
}

OUT = Path(
    "research/findings/raw/lanes/perception/"
    "visual_hierarchical_part_identity_calibration.json"
)


def validate_individual_seed(phase: str, seed: int) -> int:
    supplied = int(seed)
    if phase == "smoke":
        if supplied != SMOKE_SEED:
            raise ValueError(f"smoke requires reserved seed {SMOKE_SEED}; received {supplied}")
        return supplied
    if phase not in OPEN_PHASES:
        raise ValueError(f"phase {phase!r} is locked; only calibration is open")
    if supplied not in SEED_PARTITIONS[phase]:
        raise ValueError(
            f"{phase} cannot use seed {supplied}; allowed seeds are "
            f"{list(SEED_PARTITIONS[phase])}"
        )
    return supplied


def validate_seed_partition(phase: str, seeds: Sequence[int]) -> tuple[int, ...]:
    supplied = tuple(int(seed) for seed in seeds)
    if phase == "smoke":
        expected = (SMOKE_SEED,)
    elif phase in OPEN_PHASES:
        expected = SEED_PARTITIONS[phase]
    else:
        raise ValueError(f"phase {phase!r} is locked; only calibration is open")
    if supplied != expected:
        raise ValueError(
            f"{phase} requires exact ordered seeds {list(expected)}; received {list(supplied)}"
        )
    return supplied


def aggregate_formal_verdict(phase: str, rows: Sequence[dict]) -> str:
    if phase == "smoke":
        return "NOT-SCIENTIFIC-SMOKE"
    row_seeds = tuple(int(row["seed"]) for row in rows)
    if row_seeds != CALIBRATION_SEEDS:
        raise ValueError(
            "calibration aggregate requires exact ordered result rows "
            f"{list(CALIBRATION_SEEDS)}; received {list(row_seeds)}"
        )
    statuses = tuple(str(row["formal_verdict"]) for row in rows)
    if any(status == UNDEFINED for status in statuses):
        return UNDEFINED
    return GO if all(status == GO for status in statuses) else NO_GO


def _rf_axis(center: int, n_pos: int, width: int) -> tuple[int, ...]:
    start = max(0, min(center - width // 2, n_pos - width))
    return tuple(range(start, start + width))


def retinotopic_feature_sets(
    n_orientations: int,
    n_pos: int,
    v2_grid: int,
    rf_width: int,
) -> tuple[tuple[int, ...], ...]:
    """Map each V2 hypercolumn to one local spatial patch across orientations."""
    if not 1 <= rf_width <= n_pos or not 1 <= v2_grid <= n_pos:
        raise ValueError("require 1 <= rf-width, v2-grid <= n-pos")
    centers = np.rint(np.linspace(0, n_pos - 1, v2_grid)).astype(int)
    fields = []
    for cy in centers:
        ys = _rf_axis(int(cy), n_pos, rf_width)
        for cx in centers:
            xs = _rf_axis(int(cx), n_pos, rf_width)
            fields.append(
                tuple(
                    (orientation * n_pos + y) * n_pos + x
                    for orientation in range(n_orientations)
                    for y in ys
                    for x in xs
                )
            )
    return tuple(fields)


def scrambled_feature_sets(
    local_fields: Sequence[Sequence[int]], n_in: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Destroy spatial locality while preserving every hypercolumn's fan-in."""
    rng = np.random.default_rng(seed)
    scrambled = []
    for field in local_fields:
        local = tuple(int(index) for index in field)
        candidate = tuple(sorted(int(index) for index in rng.choice(n_in, len(local), replace=False)))
        if set(candidate) == set(local):
            candidate = tuple((index + 1) % n_in for index in candidate)
        scrambled.append(candidate)
    return tuple(scrambled)


class RetinotopicPartLayer:
    """Locally wired V1-to-V2 permanences plus per-hypercolumn FS competition."""

    def __init__(
        self,
        *,
        seed: int,
        n_orientations: int,
        n_pos: int,
        v2_grid: int,
        rf_width: int,
        part_cells: int,
        fs_cells: int,
        latency_steps: int,
        lp: float,
        ld_wi: float,
        fs_enabled: bool = True,
        receptive_field_scramble: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.n_in = int(n_orientations * n_pos * n_pos)
        self.v2_grid = int(v2_grid)
        self.n_hypercolumns = self.v2_grid**2
        self.part_cells = int(part_cells)
        self.n_out = self.n_hypercolumns * self.part_cells
        self.lp = float(lp)
        self.ld_wi = float(ld_wi)
        self.fs_enabled = bool(fs_enabled)
        self.local_fields = retinotopic_feature_sets(
            n_orientations, n_pos, v2_grid, rf_width
        )
        self.wired_fields = (
            scrambled_feature_sets(self.local_fields, self.n_in, seed * 31 + 7)
            if receptive_field_scramble
            else self.local_fields
        )

        self.b, self.ci = _build_cells_bridge(
            self.seed, self.n_in + self.n_out, coincidence=True
        )
        rng = np.random.default_rng(self.seed * 101 + 11)
        pre, post, weights = [], [], []
        for tile, field in enumerate(self.wired_fields):
            for local_cell in range(self.part_cells):
                column = tile * self.part_cells + local_cell
                for feature in field:
                    pre.append(int(self.ci[feature]))
                    post.append(int(self.ci[self.n_in + column]))
                    weights.append(float(rng.uniform(0.30, 0.55)))
        self.b.inject_explicit_wiring(
            {
                "v1_to_v2_local": {
                    "pre_indices": pre,
                    "post_indices": post,
                    "initial_weights": weights,
                    "plastic": False,
                    "coincidence_detector": True,
                    "conn_type": "ff",
                }
            }
        )
        coo = self.b._get_cached_coo()
        row = np.asarray(_host(coo.row), dtype=np.int64)
        col = np.asarray(_host(coo.col), dtype=np.int64)
        cell_to_unit = {
            int(cell): unit for unit, cell in enumerate(np.asarray(self.ci, dtype=int))
        }
        ff_feat, ff_col, ff_pos = [], [], []
        for position, (pre_cell, post_cell) in enumerate(zip(row, col)):
            source = cell_to_unit.get(int(pre_cell))
            target = cell_to_unit.get(int(post_cell))
            if (
                source is not None
                and target is not None
                and source < self.n_in
                and self.n_in <= target < self.n_in + self.n_out
            ):
                ff_feat.append(source)
                ff_col.append(target - self.n_in)
                ff_pos.append(position)
        self.ff_feat = np.asarray(ff_feat, dtype=np.int64)
        self.ff_col = np.asarray(ff_col, dtype=np.int64)
        self.ff_pos = np.asarray(ff_pos, dtype=np.int64)
        self.nsyn = len(row)
        self.selectors = tuple(
            SpikeLatencySelector(
                seed=self.seed * 1009 + tile,
                n_col=self.part_cells,
                k_win=max(1, self.part_cells // 4),
                n_fs=fs_cells,
                n_steps=latency_steps,
                wta_enabled=self.fs_enabled,
            )
            for tile in range(self.n_hypercolumns)
        )
        self.selection_calls = 0
        self.total_fired = 0

    def feedforward_permanences(self) -> np.ndarray:
        return np.asarray(_host(self.b.cp_connections.data), dtype=np.float64)[
            self.ff_pos
        ].copy()

    def feedforward_drive(self, features: set[int]) -> np.ndarray:
        active = np.zeros(self.n_in, dtype=np.float64)
        if features:
            active[list(features)] = 1.0
        data = np.asarray(_host(self.b.cp_connections.data), dtype=np.float64)
        drive = np.zeros(self.n_out, dtype=np.float64)
        np.add.at(drive, self.ff_col, active[self.ff_feat] * data[self.ff_pos])
        return drive

    def encode_all_fired(self, features: set[int]) -> set[int]:
        """Return every excitatory V2 cell firing by deadline, without truncation."""
        drive = self.feedforward_drive(features)
        fired: set[int] = set()
        for tile, selector in enumerate(self.selectors):
            start = tile * self.part_cells
            selector.select(drive[start : start + self.part_cells])
            local_fired = np.flatnonzero(selector.last_first_spike <= selector.n_steps)
            fired.update(start + int(cell) for cell in local_fired)
        self.selection_calls += 1
        self.total_fired += len(fired)
        return fired

    def learn(self, features: set[int], fired: set[int]) -> None:
        active = np.zeros(self.n_in, dtype=np.float64)
        if features:
            active[list(features)] = 1.0
        pre_last = np.zeros(self.nsyn, dtype=np.float64)
        post_now = np.zeros(self.nsyn, dtype=np.float64)
        hfac = np.ones(self.nsyn, dtype=np.float64)
        pre_last[self.ff_pos] = active[self.ff_feat]
        post_now[self.ff_pos] = np.isin(
            self.ff_col, np.fromiter((int(cell) for cell in fired), dtype=int)
        ).astype(float)
        data = np.asarray(_host(self.b.cp_connections.data), dtype=np.float64)
        updated = np.asarray(
            fused_htm_permanence_update(
                data, pre_last, post_now, hfac, self.lp, 0.0, 0.0, 1.0
            )
        )
        pre_active = np.zeros(self.nsyn, dtype=np.float64)
        pre_active[self.ff_pos] = active[self.ff_feat]
        updated = np.asarray(
            fused_htm_winner_inactive_depression(
                updated, pre_active, post_now, self.ld_wi, 0.0, 1.0
            ),
            dtype=np.float32,
        )
        self.b.cp_connections.data[:] = (
            self.b.xp.asarray(updated) if hasattr(self.b, "xp") else updated
        )

    def locality_metrics(self) -> dict[str, int | bool]:
        violations = 0
        for feature, column in zip(self.ff_feat, self.ff_col):
            tile = int(column) // self.part_cells
            violations += int(int(feature) not in set(self.local_fields[tile]))
        expected = sum(len(field) * self.part_cells for field in self.wired_fields)
        return {
            "expected_local_synapses": expected,
            "actual_feedforward_synapses": int(len(self.ff_pos)),
            "nonlocal_synapses": int(violations),
            "fan_in_preserved": int(len(self.ff_pos)) == expected,
        }

    def selection_metrics(self) -> dict[str, float | bool | str]:
        return {
            "readout": "all V2 excitatory cells firing by fixed deadline",
            "host_top_k_or_first_k_truncation": False,
            "local_fs_pathways_enabled": self.fs_enabled,
            "mean_fired_fraction": round(
                self.total_fired / max(self.selection_calls * self.n_out, 1), 6
            ),
        }


class TraceIdentityLayer(TraceV1Pooler):
    """Sparse IT spike population with presynaptic trace and no post persistence."""

    def __init__(
        self,
        *,
        seed: int,
        n_in: int,
        n_col: int,
        expected_active: int,
        fs_cells: int,
        latency_steps: int,
        lp: float,
        ld_wi: float,
        fs_enabled: bool = True,
    ) -> None:
        super().__init__(seed, n_in, n_col, expected_active, lp=lp, ld_wi=ld_wi)
        self.selector = SpikeLatencySelector(
            seed=seed * 1009 + 29,
            n_col=n_col,
            k_win=expected_active,
            n_fs=fs_cells,
            n_steps=latency_steps,
            wta_enabled=fs_enabled,
        )
        self.trace = np.zeros(n_in, dtype=np.float64)
        self.trace_only_update_synapses = 0
        self.fs_enabled = bool(fs_enabled)
        self.selection_calls = 0
        self.total_fired = 0

    def feedforward_permanences(self) -> np.ndarray:
        return np.asarray(_host(self.b.cp_connections.data), dtype=np.float64)[
            self.ff_pos
        ].copy()

    def begin_track(self) -> None:
        self.trace.fill(0.0)

    def encode_all_fired(self, features: set[int]) -> set[int]:
        self.selector.select(super()._drive(features))
        fired = set(
            int(cell)
            for cell in np.flatnonzero(
                self.selector.last_first_spike <= self.selector.n_steps
            )
        )
        self.selection_calls += 1
        self.total_fired += len(fired)
        return fired

    def learn(
        self,
        features: set[int],
        fired: set[int],
        *,
        trace_enabled: bool,
        trace_decay: float,
    ) -> None:
        current = np.zeros(self.n_in, dtype=np.float64)
        if features:
            current[list(features)] = 1.0
        self.trace = (
            current
            if not trace_enabled
            else np.clip(self.trace * trace_decay + current, 0.0, 1.0)
        )
        fired_array = np.fromiter((int(cell) for cell in fired), dtype=int)
        traced_only = (
            (self.trace[self.ff_feat] > 0.05)
            & (current[self.ff_feat] == 0.0)
            & np.isin(self.ff_col, fired_array)
        )
        self.trace_only_update_synapses += int(np.count_nonzero(traced_only))
        _apply_traced_potentiation(self, self.trace, fired, self.lp)
        self._winner_inactive_traced(fired, self.trace, self.ld_wi)

    def selection_metrics(self) -> dict[str, float | bool | str | int]:
        return {
            "readout": "all IT excitatory cells firing by fixed deadline",
            "host_top_k_or_first_k_truncation": False,
            "fs_pathways_enabled": self.fs_enabled,
            "postsynaptic_persistence_present": False,
            "trace_only_update_synapses": self.trace_only_update_synapses,
            "mean_fired_fraction": round(
                self.total_fired / max(self.selection_calls * self.n_col, 1), 6
            ),
        }


def _permanence_change(before: np.ndarray, after: np.ndarray) -> dict[str, float | int]:
    delta = np.abs(np.asarray(after) - np.asarray(before))
    return {
        "changed_synapses": int(np.count_nonzero(delta > 1e-8)),
        "changed_fraction": round(float(np.mean(delta > 1e-8)), 6),
        "mean_absolute_change": round(float(np.mean(delta)), 6),
        "permanences_live_in": "SimulationBridge.cp_connections.data",
    }


def _build_layers(seed: int, args: argparse.Namespace, spec: dict):
    v2 = RetinotopicPartLayer(
        seed=seed * 1009 + 41,
        n_orientations=args.n_orientations,
        n_pos=args.n_pos,
        v2_grid=args.v2_grid,
        rf_width=args.rf_width,
        part_cells=args.v2_part_cells,
        fs_cells=args.v2_fs_cells,
        latency_steps=args.latency_steps,
        lp=args.v2_lr_pot,
        ld_wi=args.v2_lr_depress,
        fs_enabled=bool(spec.get("v2_fs", True)),
        receptive_field_scramble=bool(spec.get("rf_scramble", False)),
    )
    it = TraceIdentityLayer(
        seed=seed * 1009 + 53,
        n_in=v2.n_out,
        n_col=args.it_cells,
        expected_active=args.it_expected_active,
        fs_cells=args.it_fs_cells,
        latency_steps=args.latency_steps,
        lp=args.it_lr_pot,
        ld_wi=args.it_lr_depress,
        fs_enabled=bool(spec.get("it_fs", True)),
    )
    return v2, it


def _infer_codes(
    v2: RetinotopicPartLayer,
    it: TraceIdentityLayer,
    v1_codes: Sequence[set[int]],
) -> np.ndarray:
    codes = []
    for v1_code in v1_codes:
        v2_code = v2.encode_all_fired(v1_code)
        codes.append(it.encode_all_fired(v2_code))
    return _binary_codes(codes, it.n_col)


def _train_arm(
    v2: RetinotopicPartLayer,
    it: TraceIdentityLayer,
    v1_codes: Sequence[set[int]],
    tracks: Sequence[Sequence[int]],
    spec: dict,
    args: argparse.Namespace,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed * 1009 + 67)
    order = np.arange(len(tracks))
    for _ in range(args.epochs):
        rng.shuffle(order)
        for track_index in order:
            it.begin_track()
            for frame_index in tracks[int(track_index)]:
                v1_code = v1_codes[int(frame_index)]
                v2_code = v2.encode_all_fired(v1_code)
                it_code = it.encode_all_fired(v2_code)
                if spec["v2_learning"]:
                    v2.learn(v1_code, v2_code)
                if spec["it_learning"]:
                    it.learn(
                        v2_code,
                        it_code,
                        trace_enabled=bool(spec["trace"]),
                        trace_decay=args.trace_decay,
                    )


def scientific_diagnostics(arms: dict, pixel_scramble: dict) -> dict[str, bool]:
    intact = arms["intact"]
    off = arms["all_learning_off"]
    v2_off = arms["v2_learning_off"]
    trace_off = arms["it_trace_off"]
    shuffled = arms["temporal_shuffle"]
    v2_fs = arms["v2_local_fs_lesion"]
    rf_scramble = arms["receptive_field_scramble"]
    return {
        "intact_decode_at_least_0_60": intact["heldout_identity_decode"] >= 0.60,
        "intact_margin_at_least_0_10": intact["held_to_train_cosine_margin"] >= 0.10,
        "all_learning_off_costs_0_20": (
            intact["heldout_identity_decode"] >= off["heldout_identity_decode"] + 0.20
        ),
        "v2_learning_is_load_bearing": (
            intact["heldout_identity_decode"] >= v2_off["heldout_identity_decode"] + 0.10
        ),
        "it_trace_is_load_bearing": (
            intact["heldout_identity_decode"] >= trace_off["heldout_identity_decode"] + 0.10
        ),
        "temporal_order_is_load_bearing": (
            intact["heldout_identity_decode"] >= shuffled["heldout_identity_decode"] + 0.15
        ),
        "v2_fs_raises_density": (
            v2_fs["v2_selection"]["mean_fired_fraction"]
            >= intact["v2_selection"]["mean_fired_fraction"] + 0.20
        ),
        "v2_fs_costs_decode": (
            intact["heldout_identity_decode"] >= v2_fs["heldout_identity_decode"] + 0.10
        ),
        "rf_scramble_at_most_0_35": rf_scramble["heldout_identity_decode"] <= 0.35,
        "rf_scramble_costs_0_20": (
            intact["heldout_identity_decode"] >= rf_scramble["heldout_identity_decode"] + 0.20
        ),
        "pixel_scramble_at_most_0_35": pixel_scramble["heldout_identity_decode"] <= 0.35,
        "pixel_scramble_costs_0_20": (
            intact["heldout_identity_decode"] >= pixel_scramble["heldout_identity_decode"] + 0.20
        ),
        "both_intact_learning_stages_change": (
            intact["v2_learning"]["changed_synapses"] > 0
            and intact["it_learning"]["changed_synapses"] > 0
        ),
        "no_learning_changes_zero": (
            off["v2_learning"]["changed_synapses"] == 0
            and off["it_learning"]["changed_synapses"] == 0
        ),
    }


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    validate_individual_seed(args.phase, seed)
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
        receptive_field_radius=args.gabor_rf_radius,
    )
    train_activity = dense_v1_activity(dataset.train_images, args, response_matrix)
    held_activity = dense_v1_activity(dataset.held_images, args, response_matrix)
    scrambled_activity = dense_v1_activity(
        _scramble_images(dataset.held_images, seed * 101 + 79), args, response_matrix
    )
    n_v1 = args.n_orientations * args.n_pos * args.n_pos
    v1_encoder = SpikingV1Encoder(
        seed=seed * 1009 + 17,
        n_features=n_v1,
        k_active=args.v1_active,
        n_fs=args.v1_fs_cells,
        n_steps=args.latency_steps,
    )
    train_v1 = v1_encoder.encode_many(train_activity)
    held_v1 = v1_encoder.encode_many(held_activity)
    scrambled_v1 = v1_encoder.encode_many(scrambled_activity)
    shuffled = shuffled_track_indices(dataset.train_tracks, seed * 101 + 83)

    arms: dict[str, dict] = {}
    intact_layers = None
    intact_train_codes = None
    for arm_name, base_spec in ARM_SPECS.items():
        spec = {
            "v2_learning": False,
            "it_learning": False,
            "trace": False,
            "temporal_shuffle": False,
            "v2_fs": True,
            "it_fs": True,
            "rf_scramble": False,
            **base_spec,
        }
        v2, it = _build_layers(seed, args, spec)
        v2_before = v2.feedforward_permanences()
        it_before = it.feedforward_permanences()
        tracks = shuffled if spec["temporal_shuffle"] else dataset.train_tracks
        _train_arm(v2, it, train_v1, tracks, spec, args, seed)
        v2_after = v2.feedforward_permanences()
        it_after = it.feedforward_permanences()
        train_codes = _infer_codes(v2, it, train_v1)
        held_codes = _infer_codes(v2, it, held_v1)
        arms[arm_name] = {
            **score_codes(
                train_codes,
                dataset.train_object_ids,
                held_codes,
                dataset.held_object_ids,
            ),
            "v2_learning": _permanence_change(v2_before, v2_after),
            "it_learning": _permanence_change(it_before, it_after),
            "v2_selection": v2.selection_metrics(),
            "it_selection": it.selection_metrics(),
            "v2_connectivity": v2.locality_metrics(),
            "mechanism": spec,
        }
        if arm_name == "intact":
            intact_layers = (v2, it)
            intact_train_codes = train_codes

    assert intact_layers is not None and intact_train_codes is not None
    intact_v2, intact_it = intact_layers
    scramble_codes = _infer_codes(intact_v2, intact_it, scrambled_v1)
    pixel_scramble = score_codes(
        intact_train_codes,
        dataset.train_object_ids,
        scramble_codes,
        dataset.held_object_ids,
    )
    diagnostics = scientific_diagnostics(arms, pixel_scramble)

    exact_frame_multiset = sorted(
        index for track in dataset.train_tracks for index in track
    ) == sorted(index for track in shuffled for index in track)
    all_scientific = set().union(*(set(values) for values in SEED_PARTITIONS.values()))
    stream_checks = {
        "smoke_seed_outside_scientific_partitions": SMOKE_SEED not in all_scientific,
        "scientific_partitions_disjoint": all(
            set(left).isdisjoint(set(right))
            for i, left in enumerate(SEED_PARTITIONS.values())
            for right in list(SEED_PARTITIONS.values())[i + 1 :]
        ),
        "temporal_shuffle_preserves_exact_frame_multiset": exact_frame_multiset,
        "labels_enter_encoding_or_learning": False,
        "v2_code_uses_all_fired_cells": True,
        "v2_host_top_k_or_first_k_truncation": False,
        "postsynaptic_persistence_present": False,
        "intact_v2_has_no_nonlocal_synapses": (
            arms["intact"]["v2_connectivity"]["nonlocal_synapses"] == 0
        ),
        "rf_scramble_preserves_synapse_count": (
            arms["intact"]["v2_connectivity"]["actual_feedforward_synapses"]
            == arms["receptive_field_scramble"]["v2_connectivity"][
                "actual_feedforward_synapses"
            ]
        ),
        "fixed_scaffolds": [
            "fixed Gabor filters",
            "host V1 normalization and overlap-to-current scaling",
            "fixed V2 receptive-field topology",
            "host-maintained presynaptic IT trace",
            "synthetic track boundaries",
            "host spike-deadline readout",
            "labels used only for scoring",
        ],
    }
    smoke_checks = {
        "local_connectivity_only": stream_checks["intact_v2_has_no_nonlocal_synapses"],
        "local_v2_permanences_change": (
            arms["intact"]["v2_learning"]["changed_synapses"] > 0
        ),
        "it_permanences_change": arms["intact"]["it_learning"]["changed_synapses"] > 0,
        "no_learning_changes_zero": diagnostics["no_learning_changes_zero"],
        "prior_frame_trace_affects_it_update": (
            arms["intact"]["it_selection"]["trace_only_update_synapses"] > 0
        ),
        "v2_fs_lesion_increases_density": (
            arms["v2_local_fs_lesion"]["v2_selection"]["mean_fired_fraction"]
            > arms["intact"]["v2_selection"]["mean_fired_fraction"]
        ),
        "labels_absent": not stream_checks["labels_enter_encoding_or_learning"],
        "v2_all_fired_readout": stream_checks["v2_code_uses_all_fired_cells"],
    }

    validity = Verdict("visual hierarchical part identity", chance=0.25)
    validity.require(
        "smoke seed is outside scientific partitions",
        stream_checks["smoke_seed_outside_scientific_partitions"],
        expect=True,
    )
    validity.require(
        "scientific partitions are disjoint",
        stream_checks["scientific_partitions_disjoint"],
        expect=True,
    )
    validity.require(
        "temporal shuffle preserves exact frame multiset",
        stream_checks["temporal_shuffle_preserves_exact_frame_multiset"],
        expect=True,
    )
    validity.require(
        "labels do not enter encoding or learning",
        stream_checks["labels_enter_encoding_or_learning"],
        expect=False,
    )
    validity.require(
        "intact V2 contains no nonlocal feedforward synapses",
        stream_checks["intact_v2_has_no_nonlocal_synapses"],
        expect=True,
    )
    decided = validity.decide(go=all(diagnostics.values()), verbose=False)
    formal = (
        "NOT-SCIENTIFIC-SMOKE" if args.phase == "smoke" else decided["status"]
    )
    return {
        "seed": seed,
        "phase": args.phase,
        "scientific_evidence": args.phase != "smoke",
        "formal_verdict": formal,
        "calibration_status": (
            "SMOKE-ONLY"
            if args.phase == "smoke"
            else (
                "UNDEFINED"
                if decided["status"] == UNDEFINED
                else "CALIBRATION_PASS" if decided["go"] else "CALIBRATION_FAIL"
            )
        ),
        "arms": arms,
        "pixel_scramble": pixel_scramble,
        "scientific_diagnostics": diagnostics,
        "smoke_checks": smoke_checks,
        "stream_checks": stream_checks,
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
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
    parser.add_argument("--gabor-rf-radius", type=int, default=4)
    parser.add_argument("--complex-norm", default="local_orient_div")
    parser.add_argument("--v1-active", type=int, default=32)
    parser.add_argument("--v1-fs-cells", type=int, default=12)
    parser.add_argument("--v2-grid", type=int, default=4)
    parser.add_argument("--rf-width", type=int, default=3)
    parser.add_argument("--v2-part-cells", type=int, default=12)
    parser.add_argument("--v2-fs-cells", type=int, default=4)
    parser.add_argument("--it-cells", type=int, default=128)
    parser.add_argument("--it-expected-active", type=int, default=12)
    parser.add_argument("--it-fs-cells", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trace-decay", type=float, default=TRACE_DECAY)
    parser.add_argument("--v2-lr-pot", type=float, default=0.05)
    parser.add_argument("--v2-lr-depress", type=float, default=0.01)
    parser.add_argument("--it-lr-pot", type=float, default=0.05)
    parser.add_argument("--it-lr-depress", type=float, default=0.01)
    parser.add_argument("--latency-steps", type=int, default=40)
    parser.add_argument("--out", default=str(OUT))
    return parser


def validate_config(args: argparse.Namespace) -> None:
    n_v1 = args.n_orientations * args.n_pos * args.n_pos
    if not 0 < args.v1_active < n_v1:
        raise ValueError("v1-active must be between zero and the V1 feature count")
    if not 1 <= args.rf_width <= args.n_pos or not 1 <= args.v2_grid <= args.n_pos:
        raise ValueError("rf-width and v2-grid must be in [1, n-pos]")
    if not 0 < args.it_expected_active < args.it_cells:
        raise ValueError("it-expected-active must be between zero and it-cells")
    if min(args.v2_part_cells, args.v2_fs_cells, args.it_fs_cells, args.latency_steps) < 1:
        raise ValueError("population sizes and latency steps must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_seed_partition(args.phase, args.seeds)
    validate_config(args)
    started = time.time()
    rows = []
    for seed in args.seeds:
        row = run_seed(seed, args)
        rows.append(row)
        print(
            f"[seed {seed}] phase={args.phase} status={row['calibration_status']} "
            f"smoke={sum(row['smoke_checks'].values())}/{len(row['smoke_checks'])}",
            flush=True,
        )
    output = {
        "summary": {
            "probe": "laneD_visual_hierarchical_part_identity_gate",
            "phase": args.phase,
            "seeds": args.seeds,
            "formal_verdict": aggregate_formal_verdict(args.phase, rows),
            "scientific_evidence": args.phase != "smoke",
            "seed_partitions": {
                name: list(values) for name, values in SEED_PARTITIONS.items()
            },
            "smoke_seed": SMOKE_SEED,
            "controls": list(ARM_SPECS) + ["pixel_scramble"],
            "mechanism": (
                "Retinotopic local V1-to-V2 permanences and per-hypercolumn FS/PV "
                "competition produce an untruncated all-fired V2 part code. A sparse "
                "IT population learns continuous views through a presynaptic Foldiak "
                "trace without postsynaptic persistence."
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
