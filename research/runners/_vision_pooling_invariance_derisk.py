"""Position-invariant object recognition via complex-cell POOLING + Foldiak/SFA temporal-continuity.

Board task #44 ("recognize an object wherever it appears"). Two documented NO-GOs precede this on
the same V1 front end: sharper feature COMPETITION (2026-08-11 harder-kWTA) and a learned
DECORRELATION stage (2026-08-18 laneD-v1-pooler-learned-decorrelation-NOGO). Both "separate
features but do not POOL across position." The named missing mechanism is a complex-cell POOLING
operation (Hubel-Wiesel): pool one feature's responses ACROSS a spread of positions into a
position-invariant unit -- LEARNED, not hand-set, with temporal continuity as the teacher (Foldiak
1991; research/biology/invariance-from-temporal-continuity.md).

THIS RUNNER IS A DECISIVE MULTI-ARM PROBE. It does not just try one pooler; it decomposes the
invariance into (topology) vs (learning) with a shared V1 front end and one GO gate, over 6 seeds.

ARMS (all read the deployed Gabor/V1 -> V1-complex(orient x pos) -> local orientation competition):
  A  V1-DIRECT            no pooling (baseline; the position-specific code the 2 NO-GOs also read).
  B  LEARNED-GLOBAL       the target emergent mechanism: n_units units, each g_i = W_i . x over the
                          FULL feature vector; trace-based competition (winner on the leaky-
                          integrated activity) with duty-cycle boosting; the cross-position pooling
                          TOPOLOGY is learned from a moving-object continuity stream by the Foldiak
                          trace rule. Controls: B_shuffled (same frames, order destroyed) and
                          B_frozen (random W, no learning).
  C  SCAFFOLD-LOCAL       innate LOCAL retinotopic pooling windows (a FLAGGED host scaffold =
                          developmental complex-cell RFs); within each window the trace rule learns
                          the orientation preference. Controls: C_frozen (random prefs), C_shuffled.
  D  ORACLE-GLOBAL        hand-wired sum over ALL positions per orientation channel (the pooling
                          ceiling / headroom reference; fully host-designed).

ANTI-CHEATS (they ARE the result):
  1. HELD-OUT POSITIONS. Learn on train positions; decode at interleaved positions NEVER seen in
     training. Invariance means held-position decode >> V1-direct.
  2. TEMPORAL CONTINUITY LOAD-BEARING. Shuffle the moving-stream order -> if invariance survives,
     it is a STATIC artifact of the pooling topology, not the trace rule. Reported per arm.
  3. POSITION POOLED OUT. Object decodable while position NOT decodable off the same pooled units.
  4. 6 seeds (42/43/44/100/101/102), pooled + per-seed, plus frozen and label-shuffle nulls.

GO gate (per seed, for the EMERGENT mechanism, ARM B): held-object decode >= chance+decode_margin
AND beats V1-direct, B_shuffled, and B_frozen by beat_margin AND position pooled out AND pixel-
scramble does not decode. The scaffold arm C is scored separately for whether LEARNING (not the
innate window) carries the invariance: C_trace must beat C_frozen by beat_margin.

BRAIN-BASED status: a cheap-first RATE de-risk of a spiking complex-cell layer (unit drive =
synaptic integration W.x; competition = lateral inhibition; activity trace = a slow post-synaptic
Ca2+/eligibility variable; update = a local trace-modulated Hebbian rule). No sim/ edit; reuses the
deployed Gabor/V1 front end and the laneD bar renderer by import. A rate model is GENEROUS (if rate
fails, spiking will not save it). Any innate pooling window is a FLAGGED scaffold, per the task's
"if you must scaffold the pooling topology, flag it and burn it down toward learned."

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._vision_pooling_invariance_derisk \
      --seeds 42 --out research/findings/raw/lanes/perception/vpool_smoke.json
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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from research.runners._laneD_v1_pooler_trace_invariance_derisk import _render_bar  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_pooling_invariance.json")


# ============================================================================================
# Task + V1 front end.
# ============================================================================================
def _positions(n: int, image_size: int, span: float, axis: str) -> list[tuple[float, float]]:
    ctr = image_size * 0.5
    offs = np.linspace(-span, span, n)
    if axis == "x":
        return [(float(ctr + o), float(ctr)) for o in offs]
    return [(float(ctr), float(ctr + o)) for o in offs]


def _build_objects(cats_theta, positions, n_ex, image_size, bar_len_frac, seed, pixel_noise):
    rng = np.random.default_rng(seed)
    imgs, cat, pos = [], [], []
    base_len = image_size * bar_len_frac
    for ci, th0 in enumerate(cats_theta):
        for pi, (cx0, cy0) in enumerate(positions):
            for _ in range(n_ex):
                th = th0 + rng.normal(0.0, math.radians(5.0))
                cx = cx0 + rng.normal(0.0, image_size * 0.012)
                cy = cy0 + rng.normal(0.0, image_size * 0.012)
                ln = base_len * (1.0 + rng.normal(0.0, 0.05))
                tk = 1.6 * (1.0 + rng.normal(0.0, 0.08))
                imgs.append(_render_bar(cx, cy, th, ln, tk, rng, image_size, pixel_noise))
                cat.append(ci)
                pos.append(pi)
    return (np.asarray(imgs, dtype=np.float32), np.asarray(cat, np.int64), np.asarray(pos, np.int64))


def _scramble_images(images, seed):
    rng = np.random.default_rng(seed)
    c, h, w = images.shape[1:]
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        perm = rng.permutation(h * w)
        out[i] = images[i].reshape(c, h * w)[:, perm].reshape(c, h, w)
    return out


def _orient_competition(codes, mode, n_orient, n_pos):
    """Local orientation competition at each retinotopic position (lateral inhibition among
    orientation columns of a hypercolumn). NOT the pooling stage: it cleans the per-position
    orientation signal and does nothing across position. Applied identically to every arm."""
    if mode == "none":
        return codes.astype(np.float32)
    n_pos2 = n_pos * n_pos
    m = codes.reshape(codes.shape[0], n_orient, n_pos2).astype(np.float64, copy=True)
    eps = 1e-6
    if mode == "div":
        m = m / (m.sum(axis=1, keepdims=True) + eps)
    elif mode == "z":
        mu = m.mean(axis=1, keepdims=True)
        sd = m.std(axis=1, keepdims=True)
        m = np.maximum((m - mu) / (sd + eps), 0.0)
    else:
        raise ValueError(mode)
    return m.reshape(codes.shape[0], -1).astype(np.float32)


# ============================================================================================
# ARM B: learned GLOBAL trace pool (cross-position pooling TOPOLOGY learned from scratch).
# ============================================================================================
class GlobalTracePool:
    """Foldiak (1991) trace rule with a POOLING readout, learned over the FULL feature vector.
    Unit i drive g_i = W_i . x (W_i >= 0, L2-normalised). Winner is chosen on the leaky-integrated
    activity y_bar (trace-based competition -> hysteresis that tracks the moving object across
    positions). Duty-cycle boosting spreads units over objects. Trace-modulated Hebbian:
    dW_ij = lr * y_bar_i * x_j, then renormalise. lr=0 => frozen random pool (no-learning null)."""

    def __init__(self, n_in, n_units, seed):
        self.n_in, self.n_units = int(n_in), int(n_units)
        rng = np.random.default_rng(seed)
        self.W = self._renorm(rng.random((self.n_units, self.n_in)) + 0.01)

    @staticmethod
    def _renorm(W):
        W = np.maximum(W, 0.0)
        n = np.linalg.norm(W, axis=1, keepdims=True)
        return W / np.where(n < 1e-12, 1.0, n)

    def train(self, stream, epochs, lr, decay, boost_beta):
        if lr <= 0.0:
            return
        T = max(len(stream), 1)
        duty = np.ones(self.n_units) / self.n_units
        for _ in range(int(epochs)):
            y_bar = np.zeros(self.n_units)
            wins = np.zeros(self.n_units)
            boost = np.exp(boost_beta * (1.0 / self.n_units - duty))
            for x in stream:
                g = (self.W @ x) * boost
                y_bar = decay * y_bar + (1.0 - decay) * g
                k = int(np.argmax(y_bar))
                wins[k] += 1.0
                self.W[k] += lr * y_bar[k] * x
                nn = np.linalg.norm(self.W[k])
                self.W[k] = np.maximum(self.W[k], 0.0) / (nn if nn > 1e-12 else 1.0)
            duty = 0.5 * duty + 0.5 * (wins / T)

    def pool(self, X):
        return (X @ self.W.T).astype(np.float32)


# ============================================================================================
# ARM C: innate LOCAL retinotopic pooling windows (FLAGGED scaffold) + trace-learned prefs.
# ============================================================================================
class LocalScaffoldPool:
    """Innate LOCAL retinotopic pooling: units tile the movement axis in overlapping windows (a
    FLAGGED developmental complex-cell RF scaffold). Within each window the trace rule learns an
    orientation-preference vector (slots per window, competing). Pooling = sum over the window's
    positions of the learned orientation weighting. frozen=True => random prefs (isolates whether
    LEARNING, not the innate window, carries the invariance)."""

    def __init__(self, n_orient, n_pos, win, stride, slots, seed):
        self.n_orient, self.n_pos = int(n_orient), int(n_pos)
        self.win, self.stride, self.slots = int(win), int(stride), int(slots)
        self.starts = list(range(0, self.n_pos - self.win + 1, self.stride)) or [0]
        self.nwin = len(self.starts)
        rng = np.random.default_rng(seed)
        O = rng.random((self.nwin, self.slots, self.n_orient)) + 0.01
        self.O = O / np.linalg.norm(O, axis=2, keepdims=True)

    def _poolvec(self, feats):
        N = feats.shape[0]
        F = feats.reshape(N, self.n_orient, self.n_pos, self.n_pos)
        pv = np.zeros((N, self.nwin, self.n_orient))
        for wi, s in enumerate(self.starts):
            pv[:, wi, :] = F[:, :, :, s:s + self.win].sum(axis=(2, 3))
        return pv

    def train(self, feats, cat, pos, n_categories, epochs_of_bouts, decay, lr, seed):
        pv = self._poolvec(feats)
        rng = np.random.default_rng(seed)
        by_cat = {c: np.where(cat == c)[0] for c in range(n_categories)}
        for wi in range(self.nwin):
            y_bar = np.zeros(self.slots)
            for _ in range(int(epochs_of_bouts)):
                cs = list(range(n_categories))
                rng.shuffle(cs)
                for c in cs:
                    idx = by_cat[c]
                    perm = rng.permutation(len(idx))
                    order = idx[perm][np.argsort(pos[idx[perm]], kind="stable")]
                    for i in order:
                        x = pv[i, wi, :]
                        g = self.O[wi] @ x
                        y_bar = decay * y_bar + (1.0 - decay) * g
                        k = int(np.argmax(y_bar))
                        self.O[wi, k] += lr * y_bar[k] * x
                        nn = np.linalg.norm(self.O[wi, k])
                        self.O[wi, k] = np.maximum(self.O[wi, k], 0.0) / (nn if nn > 1e-12 else 1.0)

    def pool(self, feats):
        pv = self._poolvec(feats)
        out = np.zeros((feats.shape[0], self.nwin * self.slots), dtype=np.float32)
        for wi in range(self.nwin):
            out[:, wi * self.slots:(wi + 1) * self.slots] = pv[:, wi, :] @ self.O[wi].T
        return out


def _oracle_global_pool(feats, n_orient, n_pos):
    """ARM D: hand-wired complex cell -- sum over ALL positions per orientation channel. The
    pooling ceiling; fully host-designed topology (the invariance headroom reference)."""
    N = feats.shape[0]
    return feats.reshape(N, n_orient, n_pos * n_pos).sum(axis=2).astype(np.float32)


# ============================================================================================
# Streams + decode.
# ============================================================================================
def _continuity_stream(feats, cat, pos, n_categories, passes, seed):
    """Moving-object stream: each bout = one object sweeping its train positions IN ORDER."""
    rng = np.random.default_rng(seed)
    by_cat = {c: np.where(cat == c)[0] for c in range(n_categories)}
    frames = []
    for _ in range(passes):
        cs = list(range(n_categories))
        rng.shuffle(cs)
        for c in cs:
            idx = by_cat[c]
            perm = rng.permutation(len(idx))
            order = idx[perm][np.argsort(pos[idx[perm]], kind="stable")]
            for i in order:
                frames.append(feats[i])
    return np.asarray(frames, dtype=np.float32)


def _cos_normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-9, 1.0, n)


def _centroid_decode(train_codes, train_labels, test_codes, test_labels):
    train = _cos_normalize(train_codes)
    test = _cos_normalize(test_codes)
    classes = np.unique(train_labels)
    cent = {}
    for c in classes:
        v = train[train_labels == c].mean(axis=0)
        nv = np.linalg.norm(v)
        cent[int(c)] = v / nv if nv > 1e-9 else v
    correct = 0
    for i in range(test.shape[0]):
        pred = max(classes, key=lambda c: float(test[i] @ cent[int(c)]))
        correct += int(pred == test_labels[i])
    return float(correct / max(1, test.shape[0]))


def _within_split_decode(codes, labels, seed):
    """Fair held-set split decode (for object-vs-position dissociation off the SAME codes)."""
    n = codes.shape[0]
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    h = n // 2
    return _centroid_decode(codes[idx[:h]], labels[idx[:h]], codes[idx[h:]], labels[idx[h:]])


# ============================================================================================
def run_seed(seed, a):
    positions = _positions(a.n_pos_total, a.image_size, a.pos_span, a.position_axis)
    held_pi = list(range(1, a.n_pos_total, 2))
    train_pi = [p for p in range(a.n_pos_total) if p not in held_pi]
    train_positions = [positions[p] for p in train_pi]
    held_positions = [positions[p] for p in held_pi]
    cats_theta = [(c / a.n_categories) * math.pi for c in range(a.n_categories)]

    tr_imgs, tr_cat, tr_pos = _build_objects(
        cats_theta, train_positions, a.n_ex, a.image_size, a.bar_len_frac, seed * 101 + 1, a.pixel_noise)
    he_imgs, he_cat, he_pos = _build_objects(
        cats_theta, held_positions, a.n_ex, a.image_size, a.bar_len_frac, seed * 101 + 2, a.pixel_noise)
    sc_imgs = _scramble_images(he_imgs, seed * 101 + 3)

    W = build_gabor_response_matrix(
        n_orientations=a.n_orientations, n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos, retina_size=a.image_size, receptive_field_radius=a.rf_radius)

    def feats(imgs):
        cx = pool_v1_to_complex(encode_v1(imgs, W), a.n_orientations, a.n_frequencies, a.n_pos)
        return _orient_competition(cx, a.orient_norm, a.n_orientations, a.n_pos)

    tr_f, he_f, sc_f = feats(tr_imgs), feats(he_imgs), feats(sc_imgs)
    n_in = tr_f.shape[1]
    chance = 1.0 / a.n_categories
    chance_pos = 1.0 / len(held_pi)

    # ---- streams (built ONLY from train-position frames; held positions never seen) ----
    grouped = _continuity_stream(tr_f, tr_cat, tr_pos, a.n_categories, a.epochs_of_bouts, seed * 17 + 5)
    shuffled = grouped.copy()
    np.random.default_rng(seed * 19 + 7).shuffle(shuffled)

    # ================= ARM A: V1-direct =================
    A_v1 = _centroid_decode(tr_f, tr_cat, he_f, he_cat)

    # ================= ARM B: learned GLOBAL trace pool (from-scratch topology) =================
    def fit_global(stream, lr):
        p = GlobalTracePool(n_in, a.n_units, seed)
        p.train(stream, a.epochs, lr, a.trace_decay, a.boost_beta)
        return p
    B_grp_pool = fit_global(grouped, a.lr)
    B_shf_pool = fit_global(shuffled, a.lr)
    B_frz_pool = fit_global(grouped, 0.0)
    B_grouped = _centroid_decode(B_grp_pool.pool(tr_f), tr_cat, B_grp_pool.pool(he_f), he_cat)
    B_shuffled = _centroid_decode(B_shf_pool.pool(tr_f), tr_cat, B_shf_pool.pool(he_f), he_cat)
    B_frozen = _centroid_decode(B_frz_pool.pool(tr_f), tr_cat, B_frz_pool.pool(he_f), he_cat)
    B_scramble = _centroid_decode(B_grp_pool.pool(tr_f), tr_cat, B_grp_pool.pool(sc_f), he_cat)

    # ================= ARM C: innate LOCAL scaffold pool + trace-learned prefs =================
    def fit_local(learn, stream_shuffle=False):
        p = LocalScaffoldPool(a.n_orientations, a.n_pos, a.win, a.stride, a.slots, seed)
        if learn:
            cat, pos = tr_cat, tr_pos
            if stream_shuffle:
                # destroy continuity: randomise the per-object position order the trace sees
                pos = np.random.default_rng(seed * 23 + 3).permutation(tr_pos)
            p.train(tr_f, cat, pos, a.n_categories, a.epochs_of_bouts, a.trace_decay, a.lr, seed * 29 + 9)
        return p
    C_trace = fit_local(True)
    C_frozen = fit_local(False)
    C_shuffle = fit_local(True, stream_shuffle=True)
    C_trace_dec = _centroid_decode(C_trace.pool(tr_f), tr_cat, C_trace.pool(he_f), he_cat)
    C_frozen_dec = _centroid_decode(C_frozen.pool(tr_f), tr_cat, C_frozen.pool(he_f), he_cat)
    C_shuffle_dec = _centroid_decode(C_shuffle.pool(tr_f), tr_cat, C_shuffle.pool(he_f), he_cat)

    # ================= ARM D: oracle global-per-orientation pool (headroom) =================
    D_oracle = _centroid_decode(
        _oracle_global_pool(tr_f, a.n_orientations, a.n_pos), tr_cat,
        _oracle_global_pool(he_f, a.n_orientations, a.n_pos), he_cat)

    # ---- anti-cheat 3: dissociation off the invariant (scaffold) code ----
    C_he_code = C_frozen.pool(he_f)  # the code that achieves invariance (topology-carried)
    obj_split = _within_split_decode(C_he_code, he_cat, seed * 31 + 11)
    pos_split = _within_split_decode(C_he_code, he_pos, seed * 31 + 13)
    position_pooled_out = (obj_split >= chance + a.decode_margin) and (pos_split <= chance_pos + a.pos_decode_margin)

    # ---- verdicts ----
    B_go = bool(
        (B_grouped >= chance + a.decode_margin)
        and (B_grouped - A_v1 >= a.beat_margin)
        and (B_grouped - B_shuffled >= a.beat_margin)
        and (B_grouped - B_frozen >= a.beat_margin)
        and (B_scramble <= chance + a.decode_margin)
        and position_pooled_out
    )
    pooling_capability = bool(D_oracle - A_v1 >= a.beat_margin)          # does pooling open invariance at all?
    scaffold_beats_v1 = bool(C_frozen_dec - A_v1 >= a.beat_margin)       # innate window alone (no learning) invariant?
    learning_load_bearing = bool(C_trace_dec - C_frozen_dec >= a.beat_margin)  # does LEARNING add over frozen?

    return {
        "seed": seed,
        "chance_object": round(chance, 4),
        "chance_position": round(chance_pos, 4),
        "train_positions_idx": train_pi,
        "held_positions_idx": held_pi,
        "n_train_images": int(tr_imgs.shape[0]),
        "n_held_images": int(he_imgs.shape[0]),
        "n_in": int(n_in),
        "held_decode": {
            "A_v1_direct": round(A_v1, 4),
            "B_learned_global_grouped": round(B_grouped, 4),
            "B_shuffled": round(B_shuffled, 4),
            "B_frozen": round(B_frozen, 4),
            "B_scramble": round(B_scramble, 4),
            "C_scaffold_local_trace": round(C_trace_dec, 4),
            "C_scaffold_local_frozen": round(C_frozen_dec, 4),
            "C_scaffold_local_shuffle": round(C_shuffle_dec, 4),
            "D_oracle_global": round(D_oracle, 4),
        },
        "dissociation": {
            "object_decode_heldsplit": round(obj_split, 4),
            "position_decode_heldsplit": round(pos_split, 4),
            "chance_object": round(chance, 4),
            "chance_position": round(chance_pos, 4),
            "position_pooled_out": position_pooled_out,
        },
        "verdicts": {
            "B_emergent_go": B_go,
            "pooling_capability": pooling_capability,
            "scaffold_topology_beats_v1": scaffold_beats_v1,
            "learning_load_bearing": learning_load_bearing,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--n-categories", type=int, default=4)
    p.add_argument("--n-pos-total", type=int, default=8)
    p.add_argument("--pos-span", type=float, default=11.0, help="+/- pixels the object centre spans.")
    p.add_argument("--position-axis", choices=["x", "y"], default="x")
    p.add_argument("--n-ex", type=int, default=3)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--bar-len-frac", type=float, default=0.35)
    p.add_argument("--pixel-noise", type=float, default=0.03)
    # V1 front end
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-frequencies", type=int, default=2)
    p.add_argument("--n-pos", type=int, default=16)
    p.add_argument("--rf-radius", type=int, default=2)
    p.add_argument("--orient-norm", choices=["none", "div", "z"], default="z")
    # ARM B (learned global)
    p.add_argument("--n-units", type=int, default=12)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--trace-decay", type=float, default=0.95)
    p.add_argument("--boost-beta", type=float, default=2.0)
    p.add_argument("--epochs-of-bouts", type=int, default=6)
    # ARM C (innate local scaffold)
    p.add_argument("--win", type=int, default=8, help="Innate pooling window over the movement axis (FLAGGED scaffold).")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--slots", type=int, default=4)
    # gate thresholds
    p.add_argument("--decode-margin", type=float, default=0.15)
    p.add_argument("--beat-margin", type=float, default=0.10)
    p.add_argument("--pos-decode-margin", type=float, default=0.15)
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    t0 = time.time()
    print(f"[vision-pooling-invariance] seeds={a.seeds} cats={a.n_categories} pos={a.n_pos_total}@span{a.pos_span} "
          f"V1(orient={a.n_orientations},pos={a.n_pos},rf={a.rf_radius},norm={a.orient_norm}) "
          f"B(units={a.n_units},decay={a.trace_decay}) C(win={a.win},stride={a.stride},slots={a.slots})", flush=True)

    rows = [run_seed(s, a) for s in a.seeds]
    for r in rows:
        d = r["held_decode"]
        di = r["dissociation"]
        v = r["verdicts"]
        print(f"  [seed {r['seed']}] V1 {d['A_v1_direct']:.2f} | B grp {d['B_learned_global_grouped']:.2f} "
              f"shf {d['B_shuffled']:.2f} frz {d['B_frozen']:.2f} | C trc {d['C_scaffold_local_trace']:.2f} "
              f"frz {d['C_scaffold_local_frozen']:.2f} | D oracle {d['D_oracle_global']:.2f} "
              f"| obj/pos {di['object_decode_heldsplit']:.2f}/{di['position_decode_heldsplit']:.2f} "
              f"| Bgo={v['B_emergent_go']} learn_lb={v['learning_load_bearing']}", flush=True)

    def mean(path):
        vals = []
        for r in rows:
            cur = r
            for k in path:
                cur = cur[k]
            vals.append(float(cur))
        return round(float(np.mean(vals)), 4)

    def frac(path):
        return round(float(np.mean([1.0 if _get(r, path) else 0.0 for r in rows])), 4)

    def _get(r, path):
        cur = r
        for k in path:
            cur = cur[k]
        return cur

    # ---- ATTRIBUTION: whose is the invariance? (the load-bearing question, not just both numbers) ----
    hd = lambda k: mean(("held_decode", k))  # noqa: E731
    # (1) The invariance that DOES appear (scaffold pool): is it attributable to LEARNING, or to the
    #     innate frozen topology? attributable_to = (trace - frozen)/trace.
    attributable_to("scaffold-pool held-invariance -> LEARNING (vs frozen topology)",
                    hd("C_scaffold_local_trace"), hd("C_scaffold_local_frozen"))
    # (2) The scaffold pool over the shared V1-direct baseline: attributable to the POOLING TOPOLOGY.
    attributable_to("scaffold-pool held-invariance -> POOLING TOPOLOGY (vs V1-direct)",
                    hd("C_scaffold_local_frozen"), hd("A_v1_direct"))
    # (3) ARM B (learned-from-scratch pool): is its held decode attributable to temporal continuity
    #     (grouped) over the shuffled control? (negative => the control exceeds it: no continuity effect.)
    attributable_to("ARM-B held decode -> TEMPORAL CONTINUITY (vs shuffled)",
                    hd("B_learned_global_grouped"), hd("B_shuffled"))

    n_bgo = sum(1 for r in rows if r["verdicts"]["B_emergent_go"])
    overall = ("POOL-INVARIANCE-GO" if n_bgo == len(rows)
               else "POOL-INVARIANCE-NOGO" if n_bgo == 0
               else f"POOL-INVARIANCE-PARTIAL-{n_bgo}/{len(rows)}")

    summary = {
        "probe": "vision_pooling_invariance",
        "overall_verdict": overall,
        "seeds": a.seeds,
        "n_seeds": len(rows),
        "chance_object": round(1.0 / a.n_categories, 4),
        "per_seed_B_emergent_go": [r["verdicts"]["B_emergent_go"] for r in rows],
        "held_decode_means": {
            "A_v1_direct": mean(("held_decode", "A_v1_direct")),
            "B_learned_global_grouped": mean(("held_decode", "B_learned_global_grouped")),
            "B_shuffled": mean(("held_decode", "B_shuffled")),
            "B_frozen": mean(("held_decode", "B_frozen")),
            "B_scramble": mean(("held_decode", "B_scramble")),
            "C_scaffold_local_trace": mean(("held_decode", "C_scaffold_local_trace")),
            "C_scaffold_local_frozen": mean(("held_decode", "C_scaffold_local_frozen")),
            "C_scaffold_local_shuffle": mean(("held_decode", "C_scaffold_local_shuffle")),
            "D_oracle_global": mean(("held_decode", "D_oracle_global")),
        },
        "dissociation_means": {
            "object_decode_heldsplit": mean(("dissociation", "object_decode_heldsplit")),
            "position_decode_heldsplit": mean(("dissociation", "position_decode_heldsplit")),
        },
        "verdict_fracs": {
            "B_emergent_go": frac(("verdicts", "B_emergent_go")),
            "pooling_capability": frac(("verdicts", "pooling_capability")),
            "scaffold_topology_beats_v1": frac(("verdicts", "scaffold_topology_beats_v1")),
            "learning_load_bearing": frac(("verdicts", "learning_load_bearing")),
        },
        "headroom": {
            "oracle_minus_v1": round(mean(("held_decode", "D_oracle_global")) - mean(("held_decode", "A_v1_direct")), 4),
            "B_learned_minus_v1": round(mean(("held_decode", "B_learned_global_grouped")) - mean(("held_decode", "A_v1_direct")), 4),
            "scaffold_frozen_minus_v1": round(mean(("held_decode", "C_scaffold_local_frozen")) - mean(("held_decode", "A_v1_direct")), 4),
            "scaffold_trace_minus_frozen": round(mean(("held_decode", "C_scaffold_local_trace")) - mean(("held_decode", "C_scaffold_local_frozen")), 4),
        },
        "mechanism": (
            "Pixels -> deployed Gabor/V1 -> V1-complex(orient x pos) -> local orientation competition -> "
            "complex-cell pooling. ARM B learns the cross-position pooling TOPOLOGY from a moving-object "
            "continuity stream (Foldiak trace rule). ARM C uses an innate LOCAL retinotopic pooling window "
            "(flagged scaffold) with trace-learned orientation prefs. ARM D is the hand-wired oracle pool."
        ),
        "go_gate": (
            "ARM B (emergent) per seed: held-object decode >= chance+decode_margin; beats V1-direct, "
            "B_shuffled, B_frozen by beat_margin; pixel-scramble does not decode; object decodable while "
            "position NOT decodable off the same units. Separately: learning_load_bearing = C_trace beats "
            "C_frozen by beat_margin (does LEARNING, not the innate window, carry the invariance?)."
        ),
        "config": vars(a),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_seed": rows}, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
