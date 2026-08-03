"""Lane D CPU follow-up: route the Foldiak trace rule through V1 -> OnSubstratePooler.

Context:
  The deployed retina->V1->V2->IT STDP hierarchy is retired/inert for lane D: it
  is dead below saturation and non-selective at saturation. The record-valid
  route is the V1 -> competitive OnSubstratePooler codon, where the Foldiak trace
  rule already has a six-seed GO in EMERGE-50.

This runner keeps the lane-D DiCarlo-style task small and CPU-cheap:
  * render oriented bars where category identity is orientation and retinal
    position is a nuisance factor;
  * encode pixels through the existing Gabor/V1 front end and phase/frequency
    pool to V1-complex features;
  * train an OnSubstratePooler with the EMERGE-50 trace rule on contiguous
    same-category position-jittered bouts;
  * score held-out-position category decode and held-to-train same-vs-cross
    similarity against V1, shuffled-temporal, no-learning, and pixel-scramble
    controls.

No sim/ edit: the pooler permanences live in the bridge, and the trace path uses
the committed fused_htm_permanence_update / fused_htm_winner_inactive_depression
kernels through the existing EMERGE-46/50 helpers.

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
      --seeds 42 --epochs 4 --n-bouts 6 --bout-len 6 --n-col 64 --k-win 6 \
      --out research/findings/raw/lanes/perception/v1_pooler_trace_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _install_h5py_stub_if_missing() -> None:
    """Let non-recording bridge probes run in lean CPU envs without h5py."""
    try:
        import h5py  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types

    class _UnavailableH5File:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("h5py is required for HDF5 recording/checkpointing")

    stub = types.ModuleType("h5py")
    stub.File = _UnavailableH5File
    stub.Group = object
    sys.modules["h5py"] = stub


_install_h5py_stub_if_missing()

from research.runners._emerge14_stageC_onbridge_learning_derisk import _host  # noqa: E402
from research.runners._emerge46_spiking_stacked_pooler_derisk import OnSubstratePooler  # noqa: E402
from research.runners._emerge50_trace_rule_derisk import (  # noqa: E402
    BOUT_LEN as E50_BOUT_LEN,
    TRACE_DECAY as E50_TRACE_DECAY,
    _apply_traced_potentiation,
)
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from sim.kernels import fused_htm_winner_inactive_depression  # noqa: E402
from tools.lab import lever  # noqa: E402


OUT = Path("research/findings/raw/lanes/perception/v1_pooler_trace_invariance.json")


def _render_bar(
    cx: float,
    cy: float,
    theta: float,
    length: float,
    thickness: float,
    rng: np.random.Generator,
    image_size: int,
    pixel_noise: float,
) -> np.ndarray:
    """Render one oriented bar into a channel-first ON/OFF image."""
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    perp = np.abs(dx * math.sin(theta) - dy * math.cos(theta))
    along = dx * math.cos(theta) + dy * math.sin(theta)
    bar = np.exp(-(perp * perp) / (2.0 * thickness * thickness))
    bar = bar * (np.abs(along) <= (length / 2.0)).astype(np.float32)

    on = bar.astype(np.float32)
    gx = np.gradient(on, axis=1)
    gy = np.gradient(on, axis=0)
    off = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    off = off / (off.max() + 1e-6) * 0.3

    on = np.clip(on + rng.normal(0.0, pixel_noise, size=on.shape).astype(np.float32), 0.0, 1.0)
    off = np.clip(off + rng.normal(0.0, pixel_noise * 0.5, size=off.shape).astype(np.float32), 0.0, 1.0)
    return np.stack([on, off], axis=0)


def _position_track(n_positions: int, image_size: int, bar_len_frac: float, axis: str) -> list[tuple[float, float]]:
    """Nearby positions along a short trajectory; final positions are held out."""
    bar_len = image_size * bar_len_frac
    margin = max(3.0, bar_len * 0.5 + 1.0)
    half_span = min(image_size * 0.26, image_size * 0.5 - margin)
    if n_positions <= 1:
        offsets = np.asarray([0.0])
    else:
        offsets = np.linspace(-half_span, half_span, n_positions)
    ctr = image_size * 0.5
    if axis == "x":
        return [(float(ctr + off), float(ctr)) for off in offsets]
    if axis == "y":
        return [(float(ctr), float(ctr + off)) for off in offsets]
    raise ValueError(f"unknown position axis: {axis}")


def _build_objects(
    categories_theta: list[float],
    positions: list[tuple[float, float]],
    n_ex: int,
    image_size: int,
    bar_len_frac: float,
    seed: int,
    pixel_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    cat_labels: list[int] = []
    pos_labels: list[int] = []
    base_len = image_size * bar_len_frac
    for ci, theta0 in enumerate(categories_theta):
        for pi, (cx0, cy0) in enumerate(positions):
            for _ in range(n_ex):
                theta = theta0 + rng.normal(0.0, math.radians(5.0))
                cx = cx0 + rng.normal(0.0, image_size * 0.015)
                cy = cy0 + rng.normal(0.0, image_size * 0.015)
                length = base_len * (1.0 + rng.normal(0.0, 0.05))
                thickness = 1.6 * (1.0 + rng.normal(0.0, 0.08))
                images.append(_render_bar(cx, cy, theta, length, thickness, rng, image_size, pixel_noise))
                cat_labels.append(ci)
                pos_labels.append(pi)
    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(cat_labels, dtype=np.int64),
        np.asarray(pos_labels, dtype=np.int64),
    )


def _scramble_images(images: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c, h, w = images.shape[1:]
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        perm = rng.permutation(h * w)
        out[i] = images[i].reshape(c, h * w)[:, perm].reshape(c, h, w)
    return out


def _top_features(codes: np.ndarray, k: int) -> list[set[int]]:
    return [set(int(i) for i in np.argsort(-row)[:k]) for row in codes]


def _normalize_complex(
    codes: np.ndarray,
    mode: str,
    n_orientations: int,
    n_pos: int,
) -> np.ndarray:
    """Runner-local V1-complex normalization before sparse top-feature selection."""
    if mode == "none":
        return codes

    n_pos2 = n_pos * n_pos
    maps = codes.reshape(codes.shape[0], n_orientations, n_pos2).astype(np.float32, copy=True)
    eps = 1e-6
    if mode == "local_orient_div":
        # Divisive normalization across orientations at each retinotopic position.
        maps = maps / (maps.sum(axis=1, keepdims=True) + eps)
    elif mode == "orient_spatial_div":
        # Equalize each orientation map's total activity across the retinal sheet.
        maps = maps / (maps.sum(axis=2, keepdims=True) + eps)
    elif mode == "local_orient_z":
        mu = maps.mean(axis=1, keepdims=True)
        sd = maps.std(axis=1, keepdims=True)
        maps = np.maximum((maps - mu) / (sd + eps), 0.0)
    elif mode == "spatial_z":
        mu = maps.mean(axis=2, keepdims=True)
        sd = maps.std(axis=2, keepdims=True)
        maps = np.maximum((maps - mu) / (sd + eps), 0.0)
    else:
        raise ValueError(f"unknown complex normalization mode: {mode}")
    return maps.reshape(codes.shape[0], -1)


def _binary_codes(feats: list[set[int]], n_dim: int) -> np.ndarray:
    out = np.zeros((len(feats), n_dim), dtype=np.float32)
    for i, fs in enumerate(feats):
        if fs:
            out[i, list(fs)] = 1.0
    return out


def _cos_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm < 1e-9, 1.0, norm)


def _centroid_decode(
    train_codes: np.ndarray,
    train_labels: np.ndarray,
    test_codes: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    train = _cos_normalize(train_codes)
    test = _cos_normalize(test_codes)
    classes = np.unique(train_labels)
    centroids = {}
    for c in classes:
        v = train[train_labels == c].mean(axis=0)
        nv = np.linalg.norm(v)
        centroids[int(c)] = v / nv if nv > 1e-9 else v
    correct = 0
    for i in range(test.shape[0]):
        pred = max(classes, key=lambda c: float(test[i] @ centroids[int(c)]))
        correct += int(pred == test_labels[i])
    return float(correct / max(1, test.shape[0]))


def _held_train_margin(
    train_codes: np.ndarray,
    train_labels: np.ndarray,
    held_codes: np.ndarray,
    held_labels: np.ndarray,
) -> tuple[float, float, float]:
    train = _cos_normalize(train_codes)
    held = _cos_normalize(held_codes)
    sims = held @ train.T
    same_vals = []
    cross_vals = []
    for i in range(held.shape[0]):
        same = train_labels == held_labels[i]
        same_vals.extend(sims[i, same].tolist())
        cross_vals.extend(sims[i, ~same].tolist())
    same_m = float(np.mean(same_vals)) if same_vals else 0.0
    cross_m = float(np.mean(cross_vals)) if cross_vals else 0.0
    return same_m, cross_m, same_m - cross_m


def _metrics(
    train_codes: np.ndarray,
    train_labels: np.ndarray,
    held_codes: np.ndarray,
    held_labels: np.ndarray,
    scramble_codes: np.ndarray,
) -> dict[str, float]:
    same, cross, margin = _held_train_margin(train_codes, train_labels, held_codes, held_labels)
    return {
        "heldpos_decode": round(_centroid_decode(train_codes, train_labels, held_codes, held_labels), 4),
        "scramble_decode": round(_centroid_decode(train_codes, train_labels, scramble_codes, held_labels), 4),
        "held_train_same_cos": round(same, 4),
        "held_train_cross_cos": round(cross, 4),
        "held_train_margin": round(margin, 4),
        "mean_code_activity": round(float(held_codes.mean()), 4),
    }


def _sdr(cells) -> set[int]:
    return set(int(c) for c in cells)


class TraceV1Pooler(OnSubstratePooler):
    """OnSubstratePooler with EMERGE-50 traced pre-activity."""

    def _winner_inactive_traced(self, win: set[int], trace_pre: np.ndarray, ld: float, thr: float = 0.05) -> None:
        active_mask = (np.asarray(trace_pre) > thr).astype(float)
        pre_active = np.zeros(self.nsyn)
        post_win = np.zeros(self.nsyn)
        pre_active[self.ff_pos] = active_mask[self.ff_feat]
        post_win[self.ff_pos] = np.isin(self.ff_col, np.fromiter((int(c) for c in win), int)).astype(float)
        data = _host(self.b.cp_connections.data).astype(np.float64)
        updated = np.asarray(
            fused_htm_winner_inactive_depression(data, pre_active, post_win, ld, 0.0, 1.0)
        ).astype(np.float32)
        self.b.cp_connections.data[:] = self.b.xp.asarray(updated) if hasattr(self.b, "xp") else updated

    def train_trace(self, stream: list[set[int]], epochs: int, trace_decay: float) -> None:
        duty = np.zeros(self.n_col)
        boost = np.ones(self.n_col)
        for e in range(epochs):
            trace = np.zeros(self.n_in)
            for feats in stream:
                x = np.zeros(self.n_in)
                x[list(feats)] = 1.0
                trace = np.clip(trace * trace_decay + x, 0.0, 1.0)
                win = self._winners(feats, boost)
                _apply_traced_potentiation(self, trace, _sdr(win), self.lp)
                self._winner_inactive_traced(win, trace, self.ld_wi)
                for c in win:
                    duty[c] += 1.0
            boost = np.exp(2.0 * (self.k_win / self.n_col - duty / ((e + 1) * max(len(stream), 1))))


def _make_stream_indices(
    labels: np.ndarray,
    pos_labels: np.ndarray,
    n_categories: int,
    n_bouts: int,
    bout_len: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    by_cat = {c: np.where(labels == c)[0] for c in range(n_categories)}
    stream: list[int] = []
    for _ in range(n_bouts):
        cat = int(rng.integers(n_categories))
        idxs = by_cat[cat]
        order = idxs[np.argsort(pos_labels[idxs], kind="stable")]
        start = int(rng.integers(len(order)))
        for j in range(bout_len):
            stream.append(int(order[(start + j) % len(order)]))
    return stream


def _codes_from_pooler(pooler: OnSubstratePooler, feats: list[set[int]]) -> np.ndarray:
    return _binary_codes([pooler.codon(f) for f in feats], pooler.n_col)


def _train_trace_pooler(
    seed: int,
    n_in: int,
    a: argparse.Namespace,
    stream_feats: list[set[int]],
) -> TraceV1Pooler:
    pooler = TraceV1Pooler(
        seed=seed,
        n_in=n_in,
        n_col=a.n_col,
        k_win=a.k_win,
        lp=a.pool_lr_pot,
        ld_wi=a.pool_lr_depress,
    )
    pooler.train_trace(stream_feats, a.epochs, a.trace_decay)
    return pooler


def run_seed(seed: int, a: argparse.Namespace) -> dict:
    n_pos_total = a.n_train_pos + a.n_held_pos
    positions = _position_track(n_pos_total, a.image_size, a.bar_len_frac, a.position_axis)
    train_positions = positions[: a.n_train_pos]
    held_positions = positions[a.n_train_pos :]
    theta_offset = math.radians(a.orientation_offset_deg)
    categories_theta = [theta_offset + (c / a.n_categories) * math.pi for c in range(a.n_categories)]

    train_imgs, train_labels, train_pos = _build_objects(
        categories_theta,
        train_positions,
        a.n_ex,
        a.image_size,
        a.bar_len_frac,
        seed * 101 + 1,
        a.pixel_noise,
    )
    held_imgs, held_labels, held_pos = _build_objects(
        categories_theta,
        held_positions,
        a.n_ex,
        a.image_size,
        a.bar_len_frac,
        seed * 101 + 2,
        a.pixel_noise,
    )
    scramble_imgs = _scramble_images(held_imgs, seed * 101 + 3)

    w = build_gabor_response_matrix(
        n_orientations=a.n_orientations,
        n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos,
        retina_size=a.image_size,
        receptive_field_radius=a.rf_radius,
    )
    train_v1 = pool_v1_to_complex(encode_v1(train_imgs, w), a.n_orientations, a.n_frequencies, a.n_pos)
    held_v1 = pool_v1_to_complex(encode_v1(held_imgs, w), a.n_orientations, a.n_frequencies, a.n_pos)
    scramble_v1 = pool_v1_to_complex(encode_v1(scramble_imgs, w), a.n_orientations, a.n_frequencies, a.n_pos)
    train_v1 = _normalize_complex(train_v1, a.complex_norm, a.n_orientations, a.n_pos)
    held_v1 = _normalize_complex(held_v1, a.complex_norm, a.n_orientations, a.n_pos)
    scramble_v1 = _normalize_complex(scramble_v1, a.complex_norm, a.n_orientations, a.n_pos)

    train_feats = _top_features(train_v1, a.t_active)
    held_feats = _top_features(held_v1, a.t_active)
    scramble_feats = _top_features(scramble_v1, a.t_active)
    n_in = int(train_v1.shape[1])
    train_v1_bin = _binary_codes(train_feats, n_in)
    held_v1_bin = _binary_codes(held_feats, n_in)
    scramble_v1_bin = _binary_codes(scramble_feats, n_in)

    grouped_idx = _make_stream_indices(train_labels, train_pos, a.n_categories, a.n_bouts, a.bout_len, seed * 17 + 5)
    shuffled_idx = list(grouped_idx)
    np.random.default_rng(seed * 19 + 7).shuffle(shuffled_idx)
    grouped_stream = [train_feats[i] for i in grouped_idx]
    shuffled_stream = [train_feats[i] for i in shuffled_idx]

    grouped_pooler = _train_trace_pooler(seed, n_in, a, grouped_stream)
    shuffled_pooler = _train_trace_pooler(seed, n_in, a, shuffled_stream)
    frozen_pooler = TraceV1Pooler(seed=seed, n_in=n_in, n_col=a.n_col, k_win=a.k_win,
                                  lp=a.pool_lr_pot, ld_wi=a.pool_lr_depress)

    grouped_train = _codes_from_pooler(grouped_pooler, train_feats)
    grouped_held = _codes_from_pooler(grouped_pooler, held_feats)
    grouped_scramble = _codes_from_pooler(grouped_pooler, scramble_feats)

    shuffled_train = _codes_from_pooler(shuffled_pooler, train_feats)
    shuffled_held = _codes_from_pooler(shuffled_pooler, held_feats)
    shuffled_scramble = _codes_from_pooler(shuffled_pooler, scramble_feats)

    frozen_train = _codes_from_pooler(frozen_pooler, train_feats)
    frozen_held = _codes_from_pooler(frozen_pooler, held_feats)
    frozen_scramble = _codes_from_pooler(frozen_pooler, scramble_feats)

    v1_metrics = _metrics(train_v1_bin, train_labels, held_v1_bin, held_labels, scramble_v1_bin)
    grouped_metrics = _metrics(grouped_train, train_labels, grouped_held, held_labels, grouped_scramble)
    shuffled_metrics = _metrics(shuffled_train, train_labels, shuffled_held, held_labels, shuffled_scramble)
    frozen_metrics = _metrics(frozen_train, train_labels, frozen_held, held_labels, frozen_scramble)

    chance = 1.0 / a.n_categories
    trace_margin_delta = grouped_metrics["held_train_margin"] - shuffled_metrics["held_train_margin"]
    pooler_v1_delta = grouped_metrics["held_train_margin"] - v1_metrics["held_train_margin"]
    decode_ok = grouped_metrics["heldpos_decode"] >= chance + a.decode_margin
    beats_shuffled = trace_margin_delta >= a.trace_delta
    beats_v1 = pooler_v1_delta >= a.v1_delta
    beats_frozen = (grouped_metrics["held_train_margin"] - frozen_metrics["held_train_margin"]) >= a.trace_delta
    scramble_collapses = grouped_metrics["scramble_decode"] <= chance + a.decode_margin
    trace_go = bool(decode_ok and beats_shuffled and beats_v1 and beats_frozen and scramble_collapses)

    return {
        "seed": seed,
        "chance": round(chance, 4),
        "n_train_images": int(train_imgs.shape[0]),
        "n_held_images": int(held_imgs.shape[0]),
        "train_positions": [[round(x, 2), round(y, 2)] for x, y in train_positions],
        "held_positions": [[round(x, 2), round(y, 2)] for x, y in held_positions],
        "v1_complex": v1_metrics,
        "v1_pooler_trace": grouped_metrics,
        "shuffled_temporal": shuffled_metrics,
        "no_learning": frozen_metrics,
        "decision": {
            "trace_margin_delta_vs_shuffled": round(float(trace_margin_delta), 4),
            "pooler_margin_delta_vs_v1": round(float(pooler_v1_delta), 4),
            "decode_ok": bool(decode_ok),
            "trace_beats_shuffled": bool(beats_shuffled),
            "trace_beats_v1": bool(beats_v1),
            "trace_beats_no_learning": bool(beats_frozen),
            "scramble_collapses": bool(scramble_collapses),
            "trace_go": trace_go,
            "verdict": "TRACE-ROUTED-GO" if trace_go else "TRACE-ROUTED-NOGO",
        },
        "stream": {
            "n_bouts": a.n_bouts,
            "bout_len": a.bout_len,
            "grouped_stream_len": len(grouped_stream),
            "same_multiset_shuffled_temporal": sorted(grouped_idx) == sorted(shuffled_idx),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-categories", type=int, default=3)
    parser.add_argument("--n-train-pos", type=int, default=4)
    parser.add_argument("--n-held-pos", type=int, default=1)
    parser.add_argument("--position-axis", choices=["x", "y"], default="x")
    parser.add_argument("--n-ex", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--bar-len-frac", type=float, default=0.35)
    parser.add_argument("--orientation-offset-deg", type=float, default=0.0)
    parser.add_argument("--pixel-noise", type=float, default=0.035)
    parser.add_argument("--n-orientations", type=int, default=8)
    parser.add_argument("--n-frequencies", type=int, default=2)
    parser.add_argument("--n-pos", type=int, default=8)
    parser.add_argument("--rf-radius", type=int, default=4)
    parser.add_argument(
        "--complex-norm",
        choices=["none", "local_orient_div", "orient_spatial_div", "local_orient_z", "spatial_z"],
        default="none",
        help="Optional V1-complex normalization before top-feature selection.",
    )
    parser.add_argument("--t-active", type=int, default=24)
    parser.add_argument("--n-col", type=int, default=120)
    parser.add_argument("--k-win", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--n-bouts", type=int, default=18)
    parser.add_argument("--bout-len", type=int, default=E50_BOUT_LEN)
    parser.add_argument("--trace-decay", type=float, default=E50_TRACE_DECAY)
    parser.add_argument("--pool-lr-pot", type=float, default=0.05)
    parser.add_argument("--pool-lr-depress", type=float, default=0.02)
    parser.add_argument("--decode-margin", type=float, default=0.10)
    parser.add_argument("--trace-delta", type=float, default=0.05)
    parser.add_argument("--v1-delta", type=float, default=0.02)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    t0 = time.time()
    print(
        "[laneD v1-pooler trace] "
        f"seeds={args.seeds} categories={args.n_categories} "
        f"train_pos={args.n_train_pos} held_pos={args.n_held_pos} "
        f"pooler={args.n_col}x{args.k_win} epochs={args.epochs} "
        f"bouts={args.n_bouts}x{args.bout_len}",
        flush=True,
    )

    rows = []
    for seed in args.seeds:
        row = run_seed(seed, args)
        rows.append(row)
        gm = row["v1_pooler_trace"]
        sm = row["shuffled_temporal"]
        vm = row["v1_complex"]
        dec = row["decision"]
        print(
            f"  [seed {seed}] pooler held-decode {gm['heldpos_decode']:.2f} "
            f"margin {gm['held_train_margin']:+.3f} | shuffled {sm['held_train_margin']:+.3f} "
            f"| V1 {vm['held_train_margin']:+.3f} | trace_go={dec['trace_go']}",
            flush=True,
        )

    trace_go_flags = [bool(r["decision"]["trace_go"]) for r in rows]
    n_go = int(sum(trace_go_flags))
    if n_go == len(rows):
        overall = "TRACE-ROUTED-GO"
    elif n_go == 0:
        overall = "TRACE-ROUTED-NOGO"
    else:
        overall = f"TRACE-ROUTED-PARTIAL-{n_go}/{len(rows)}"

    def mean(path: tuple[str, ...]) -> float:
        vals = []
        for r in rows:
            cur = r
            for p in path:
                cur = cur[p]
            vals.append(float(cur))
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "probe": "laneD_v1_pooler_trace_invariance",
        "overall_verdict": overall,
        "seeds": args.seeds,
        "per_seed_trace_go": trace_go_flags,
        "chance": round(1.0 / args.n_categories, 4),
        "v1_pooler_trace_heldpos_decode_mean": round(mean(("v1_pooler_trace", "heldpos_decode")), 4),
        "v1_pooler_trace_margin_mean": round(mean(("v1_pooler_trace", "held_train_margin")), 4),
        "shuffled_temporal_margin_mean": round(mean(("shuffled_temporal", "held_train_margin")), 4),
        "v1_complex_margin_mean": round(mean(("v1_complex", "held_train_margin")), 4),
        "no_learning_margin_mean": round(mean(("no_learning", "held_train_margin")), 4),
        "mechanism": (
            "Pixels -> existing Gabor/V1 -> V1-complex top features -> OnSubstratePooler permanences in "
            "cp_connections, trained with the EMERGE-50 Foldiak trace rule over same-category position-jittered "
            "temporal bouts. The shuffled-temporal arm uses the same multiset in randomized order."
        ),
        "go_gate": (
            "Per seed: held-position decode >= chance+decode_margin, grouped trace held-to-train margin beats "
            "shuffled-temporal by trace_delta, beats V1-complex by v1_delta, beats no-learning by trace_delta, "
            "and per-image pixel scramble does not decode above chance+decode_margin."
        ),
        "config": vars(args),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    lever(
        "trace margin vs shuffled",
        summary["shuffled_temporal_margin_mean"],
        summary["v1_pooler_trace_margin_mean"],
        required=False,
        continuous=round(summary["v1_pooler_trace_margin_mean"] - summary["shuffled_temporal_margin_mean"], 4),
    )
    lever(
        "trace margin vs no-learning",
        summary["no_learning_margin_mean"],
        summary["v1_pooler_trace_margin_mean"],
        required=False,
        continuous=round(summary["v1_pooler_trace_margin_mean"] - summary["no_learning_margin_mean"], 4),
    )
    out = {"summary": summary, "per_seed": rows}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
