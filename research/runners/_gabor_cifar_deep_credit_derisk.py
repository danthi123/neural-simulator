"""GABOR-CIFAR-k DEEP-CREDIT DE-RISK (rate-numpy reference) -- the FIRST REAL (non-synthetic) task for the validated
deep-credit mechanism, off the exhausted XOR toy.

THE PLAN (already scoped): `research/findings/2026-07-07-deep-credit-real-task-scoping.md`, recommendation #1
(Gabor-CIFAR-k). Real CIFAR-10 images -> the project's FIXED Gabor/V1 front end (host-legit = sensation; the conv/V1
stage is FIXED, only the FC head learns -> sidesteps the 2026-05-18 conv-FA trainability boundary) -> a DEPTH-REQUIRED
fully-connected deep-credit classifier, k=2-4 classes. The point: prove the biologically-plausible FEEDFORWARD
deep-credit rule learns REAL hierarchical structure where DEPTH is genuinely load-bearing.

WHY PER-LAYER ALIGNMENT, NOT ACCURACY (Trap B, the 2026-05-18 MNIST VOID -- the STRONGEST prior triangulation): on
real MNIST a WRONG-SIGN (inverted) hidden rule STILL reached ~0.95 held-out because a correctly-trained linear READOUT
over rich hidden features rescues accuracy regardless of hidden-credit correctness. So held-out accuracy is NOT a valid
deep-credit instrument. The GATED metric is PER-LAYER CREDIT-ALIGNMENT vs the fenced backprop oracle (cos of the rule's
per-layer weight update vs the oracle-backprop per-layer update), with wrong-sign FAILING that alignment metric.

REUSE-BY-IMPORT (NO `sim/` edit):
  - sim.visual_cortex.build_v1_simple_weights (the fixed Gabor/V1 bank), via
    research.runners._genfrontier_optionB_visual_similarity_derisk.{build_gabor_response_matrix, encode_v1,
    pool_v1_to_complex} -- the EXACT front end EMERGE-34 uses.
  - the CIFAR-10 cache + research/findings/raw/_download_cifar.py loader (32x32x3 natural photos -> grayscale ->
    the project's (2,H,W) ON/OFF retina via unified_agent_realobject_grounded.image_to_retina convention).
  - the deep-credit ARMS + per-layer-alignment metric + anti-cheat probes from
    research.runners._gnw_d1_spiking_bdsp_derisk: FANet (plain-FA / clean-error) / MicrocircuitBDSPNet (interneuron-
    cancelled clean apical error) / BDSPNet (Burstprop) + _train + _per_layer_alignment + _no_weight_transport*.
  - sim.dendritic_mlp.DendriticMLP -- the fenced backprop oracle (ceiling + per-layer alignment reference) + the
    1-hidden-layer floor.

STAGE 0 (the load-bearing precondition -- MEASURE depth-genuineness FIRST): on the V1-CIFAR-k features, confirm a
1-hidden-layer net UNDERFITS held-out (near the shallow ceiling) while a 2-3-hidden-layer fenced-backprop ORACLE clears
it by a clear margin (oracle >= 0.80 AND oracle >= 1-layer + a real gap). If V1-CIFAR-k is NOT depth-separating at the
chosen k / feature-count, that is reported honestly (it is the wrong k, not a GO) -- this gate is what XOR/MNIST FAILED.

STAGE 1 (the deep-credit arms, once Stage 0 passes): train the deep net with the deep-credit rules -- plain-FA (FANet),
microcircuit/clean-error (MicrocircuitBDSPNet), Burstprop (BDSPNet), and the KP learned-apical-feedback variant --
plus the fenced oracle (ceiling) and the 1-layer floor. GATED METRIC = per-layer credit-alignment vs the fenced oracle.

ANTI-CHEATS (mandatory, incl. the Trap-B defeat): (1) wrong-sign hidden rule must FAIL ALIGNMENT (not just accuracy);
(2) permuted-label -> chance; (3) 1-layer floor underfits; (4) oracle >= 0.80 ceiling; (5) no weight transport
(fixed-random Y / KP reads only local pre/post).

HONEST SCOPE: this is the numpy RATE reference (the builder's fast CPU smoke). The on-bridge spiking depth-3 multi-seed
is the CONTROLLER's decisive GPU run. NO `sim/` edit anywhere (all reuse-by-import). CPU numpy backend.

Run (1-seed smoke, k=3):
    SIM_BACKEND=numpy python -m research.runners._gabor_cifar_deep_credit_derisk --seeds 42 --classes 3 \
        --hidden 64 --epochs 60 --lr 0.5
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
# TINY matmuls -> one BLAS thread per process (oversubscription is much slower); parallelize across seeds instead.
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# --- reuse-by-import: the fenced backprop oracle (ceiling + per-layer alignment reference) + the 1-hidden floor ---
from sim.dendritic_mlp import DendriticMLP  # noqa: E402
# --- reuse-by-import: the REAL Gabor/V1 front end (the exact bank EMERGE-34 uses) ---
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix, encode_v1, pool_v1_to_complex)
# --- reuse-by-import: the deep-credit arms + per-layer alignment + no-weight-transport probes ---
from research.runners._gnw_d1_spiking_bdsp_derisk import (  # noqa: E402
    BDSPNet, MicrocircuitBDSPNet, FANet, _train, _per_layer_alignment,
    _no_weight_transport, _no_weight_transport_learned, _no_weight_transport_mc)

OUT = _REPO / "research" / "findings" / "raw" / "_gabor_cifar_deep_credit.json"

# CIFAR-10 caches (data/ is gitignored). TWO supported layouts:
#  (1) the CANONICAL pickle batches (research/findings/raw/_download_cifar.py DEST) -- preferred if present;
#  (2) the fast.ai PNG-folder layout (cifar10/train/<class>/*.png) -- a fast S3 mirror, NO pickle. Same 32x32x3
#      photos; used as the fallback when the (throttled) canonical source is unavailable.
_CIFAR_DIR = _REPO / "data" / "cifar10" / "cifar-10-batches-py"          # canonical pickle dir
_CIFAR_PNG_DIR = _REPO / "data" / "cifar10" / "cifar10"                  # fast.ai PNG-folder dir
_CIFAR_BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
# CIFAR-10 class names (index order in the official batches; == the PNG folder names).
_CIFAR_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                      "dog", "frog", "horse", "ship", "truck"]

RETINA = 32  # the project's retina size (matches build_v1_simple_weights default)


# ============================================================================================================
# Real CIFAR-10 -> the project's (2, H, W) ON/OFF retina -> fixed Gabor/V1 -> V1-complex feature vector.
# ============================================================================================================
def _load_cifar_pickle(cifar_dir=_CIFAR_DIR):
    """Load the CANONICAL CIFAR-10 pickle batches -> (imgs_uint8 (N,3,32,32), labels (N,)) or None if absent.
    SECURITY: CIFAR-10's canonical on-disk format IS pickle; these are the OFFICIAL Toronto batches downloaded by
    research/findings/raw/_download_cifar.py from the canonical source (a trusted file by construction) -- never point
    this at an untrusted pickle (arbitrary-code-execution risk)."""
    cifar_dir = Path(cifar_dir)
    if not (cifar_dir / _CIFAR_BATCHES[0]).exists():
        return None
    data_list, label_list = [], []
    for b in _CIFAR_BATCHES:
        p = cifar_dir / b
        if not p.exists():
            continue
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")  # trusted official CIFAR-10 batch only (see SECURITY note above)
        data_list.append(np.asarray(d[b"data"], dtype=np.float32))         # (10000, 3072)
        label_list.append(np.asarray(d[b"labels"], dtype=np.int64))        # (10000,)
    if not data_list:
        return None
    data = np.concatenate(data_list, 0).reshape(-1, 3, 32, 32)             # (N, 3, 32, 32) RGB
    labels = np.concatenate(label_list, 0)                                 # (N,)
    return data, labels


def _load_cifar_png(png_dir=_CIFAR_PNG_DIR, split="train"):
    """Load the fast.ai PNG-folder CIFAR-10 -> (imgs_uint8 (N,3,32,32), labels (N,)) or None if absent. Layout:
    <png_dir>/<split>/<class_name>/<id>.png (real CIFAR-10 32x32x3 photos; NO pickle -- a plain image decode). Labels
    are the _CIFAR_CLASS_NAMES index of the folder name. This is the fast-mirror fallback when the throttled canonical
    pickle source is unavailable; the images are the same natural photos."""
    from PIL import Image
    base = Path(png_dir) / split
    if not base.exists():
        return None
    data_list, label_list = [], []
    for cid, cname in enumerate(_CIFAR_CLASS_NAMES):
        cdir = base / cname
        if not cdir.exists():
            continue
        for png in sorted(cdir.glob("*.png")):
            arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)  # (32,32,3)
            data_list.append(arr.transpose(2, 0, 1))                        # -> (3,32,32) channel-first (== pickle)
            label_list.append(cid)
    if not data_list:
        return None
    return np.stack(data_list, 0), np.asarray(label_list, dtype=np.int64)   # (N,3,32,32), (N,)


# fast .npz cache for the PNG-folder path (first decode is slow; subsequent seed runs load in ms).
_CIFAR_NPZ_CACHE = _REPO / "data" / "cifar10" / "cifar10_train_cache.npz"
_RAW_MEM = None  # in-process memo so the multi-seed loop decodes/loads once


def _load_cifar_raw(cifar_dir=_CIFAR_DIR):
    """Load real CIFAR-10 -> (imgs_uint8 (N,3,32,32), labels (N,)) from the canonical pickle batches if present, else
    the fast.ai PNG-folder mirror (decoded once into a fast .npz cache). Returns None if NEITHER cache exists (the
    caller prints NOT RUNNABLE). Both sources yield the identical (N,3,32,32) RGB uint8-scale contract, so downstream
    V1 encoding is source-agnostic. Memoized in-process so the multi-seed loop loads once."""
    global _RAW_MEM
    if _RAW_MEM is not None:
        return _RAW_MEM
    pk = _load_cifar_pickle(cifar_dir)
    if pk is not None:
        _RAW_MEM = pk
        return _RAW_MEM
    # PNG fallback: use the .npz cache if present, else decode all PNGs once and cache.
    if _CIFAR_NPZ_CACHE.exists():
        with np.load(_CIFAR_NPZ_CACHE, allow_pickle=False) as z:   # allow_pickle=False: plain arrays only (safe)
            _RAW_MEM = (z["data"].astype(np.float32), z["labels"].astype(np.int64))
        return _RAW_MEM
    png = _load_cifar_png()
    if png is None:
        return None
    data, labels = png
    try:  # cache the decode (uint8 keeps it ~150MB, gitignored under data/)
        np.savez(_CIFAR_NPZ_CACHE, data=data.astype(np.uint8), labels=labels.astype(np.int64))
    except OSError:
        pass
    _RAW_MEM = (data, labels)
    return _RAW_MEM


def _image_to_retina(gray):
    """Grayscale [0,1] (H,W) -> (2,H,W) ON/OFF retina (the project's convention, from
    unified_agent_realobject_grounded.image_to_retina): ON = above-mean contrast, OFF = below-mean, normalized. This
    gives the Gabor ON(+)/OFF(-) split natural bright/dark structure -- the FIXED sensory transform (host-legit)."""
    m = float(gray.mean())
    on = np.clip(gray - m, 0.0, None)
    off = np.clip(m - gray, 0.0, None)
    mx = max(on.max(), off.max(), 1e-6)
    return np.stack([on / mx, off / mx]).astype(np.float32)                # (2, H, W)


def _v1_complex_features(imgs_gray, W):
    """(N,32,32) grayscale -> (N,2,32,32) retina -> fixed Gabor/V1-simple -> V1-COMPLEX pooled features (N, n_complex).
    Uses the EXACT reused front end (build_gabor_response_matrix / encode_v1 / pool_v1_to_complex). V1-complex
    (orientation x position, frequency-pooled) is the standard invariant read the field's fixed-front-end + FC-head
    rig uses; the FC head learns on top of it."""
    retina = np.stack([_image_to_retina(g) for g in imgs_gray])            # (N, 2, 32, 32)
    v1_simple = encode_v1(retina, W)                                       # (N, n_v1_simple) rectified
    v1_complex = pool_v1_to_complex(v1_simple)                             # (N, n_complex) frequency-pooled
    return v1_complex.astype(np.float64)


def make_task_gabor_cifar(seed, classes, n_per_class=600, feature_seed=0, input_mode="v1"):
    """Load real CIFAR-k (the first `classes` CIFAR-10 class indices, or an explicit list), V1-encode, standardize,
    disjoint 65/35 train/held-out split (== the EMERGE-1 split discipline). Returns
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta -- the SAME shape the _gnw_d1 arms consume. `Ltr/Lte` are latents for the
    (secondary, reported) emergence probe = the top informative standardized V1-complex features binarized (NOT the
    gate). Feature-count is reduced by a fixed decorrelating random projection when `feature_seed` differs so the
    depth-genuineness window can be tuned (harder = fewer/decorrelated features) without any label leakage."""
    raw = _load_cifar_raw()
    if raw is None:
        return None
    data, labels = raw
    class_ids = classes if isinstance(classes, (list, tuple)) else list(range(int(classes)))
    W = build_gabor_response_matrix()                                      # the fixed Gabor/V1 bank (built once)
    rng = np.random.default_rng(seed)
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    per_class_counts = {}
    for new_lab, cid in enumerate(class_ids):
        idx = np.where(labels == cid)[0]
        idx = idx[rng.permutation(len(idx))][:n_per_class]                 # deterministic per-seed subset
        gray = data[idx].mean(1) / 255.0                                   # RGB -> grayscale (N,32,32) in [0,1]
        if input_mode == "v1":
            feats = _v1_complex_features(gray, W)                         # (N, n_complex) -- FIXED front end (too good: linearly decodable)
        elif input_mode == "raw":
            feats = gray.reshape(len(gray), -1)                           # (N, 1024) raw grayscale pixels -- NOT linearly separable => depth genuinely required
        elif input_mode == "rawrgb":
            feats = (data[idx] / 255.0).reshape(len(idx), -1)            # (N, 3072) raw RGB pixels
        else:
            raise ValueError(f"unknown input_mode {input_mode!r} (v1|raw|rawrgb)")
        cut = int(0.65 * len(feats))                                       # disjoint 65/35 split (== EMERGE-1)
        Xtr_l.append(feats[:cut]); ytr_l.append(np.full(cut, new_lab, np.int64))
        Xte_l.append(feats[cut:]); yte_l.append(np.full(len(feats) - cut, new_lab, np.int64))
        per_class_counts[_CIFAR_CLASS_NAMES[cid]] = int(len(feats))
    Xtr = np.concatenate(Xtr_l, 0); ytr = np.concatenate(ytr_l, 0)
    Xte = np.concatenate(Xte_l, 0); yte = np.concatenate(yte_l, 0)
    # Optional fixed decorrelating random projection (feature_seed>0): reduce/rotate the V1-complex features to a
    # smaller decorrelated basis so the depth-genuineness window can be tuned (fewer features = harder = more likely
    # to need depth). LABEL-FREE (a fixed random matrix), so no leakage. feature_seed=0 => identity (raw V1-complex).
    if feature_seed and feature_seed > 0:
        frng = np.random.default_rng(feature_seed + 100003)
        n_proj = min(Xtr.shape[1], max(64, int(feature_seed)))            # feature_seed doubles as the target dim
        P = frng.standard_normal((Xtr.shape[1], n_proj)) / np.sqrt(Xtr.shape[1])
        Xtr = Xtr @ P; Xte = Xte @ P
    # Per-feature STANDARDIZATION on TRAIN statistics (instrument calibration: a sigmoid MLP under-trains on merely
    # mean-centered inputs; standardized inputs are the standard mode-AGNOSTIC requirement, applied to ALL arms).
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True)
    Xtr = (Xtr - mu) / (sd + 1e-6); Xte = (Xte - mu) / (sd + 1e-6)
    # shuffle within the train/held sets so batches mix classes (deterministic per seed)
    ptr = rng.permutation(len(ytr)); pte = rng.permutation(len(yte))
    Xtr, ytr = Xtr[ptr], ytr[ptr]; Xte, yte = Xte[pte], yte[pte]
    # (secondary/reported) emergence-probe latents: the top-8 informative standardized V1-complex features binarized.
    # Pick the features by TRAIN-set between-class variance (a label-aware SELECTION of which raw features to probe,
    # but the probe target is the raw feature VALUE, not the label -- it asks 'did the hidden rep preserve these
    # informative sensory features', analogous to EMERGE-1's XOR-latent probe). Reported only; NOT the gate.
    n_lat = min(8, Xtr.shape[1])
    cls_means = np.stack([Xtr[ytr == c].mean(0) for c in np.unique(ytr)])  # (k, d)
    between_var = cls_means.var(0)                                         # (d,) between-class variance per feature
    lat_idx = np.argsort(-between_var)[:n_lat]
    Ltr = (Xtr[:, lat_idx] > 0.0).astype(np.float64)                      # binarized informative-feature latents
    Lte = (Xte[:, lat_idx] > 0.0).astype(np.float64)
    n_in = Xtr.shape[1]
    meta = {"classes": [int(c) for c in class_ids],
            "class_names": [_CIFAR_CLASS_NAMES[c] for c in class_ids],
            "n_features": int(n_in), "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
            "per_class_total": per_class_counts, "feature_seed": int(feature_seed)}
    return (Xtr, ytr, Ltr), (Xte, yte, Lte), meta


# ============================================================================================================
# Emergence probe (secondary/reported): does a linear read-out of the FROZEN hidden rep recover the informative
# sensory latents? (== EMERGE-1's _probe_latents; here the latents are informative V1-complex features.)
# ============================================================================================================
def _hidden_rep(net, X):
    acts, _lg = net._forward(np.asarray(X, float))
    return np.asarray(acts[-1])                                            # (m, last_hidden)


def _probe_latents(H_tr, L_tr, H_te, L_te):
    Xtr = np.concatenate([H_tr, np.ones((len(H_tr), 1))], 1)
    Xte = np.concatenate([H_te, np.ones((len(H_te), 1))], 1)
    lam = 1e-2 * np.eye(Xtr.shape[1]); lam[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ L_tr)
    pred = (Xte @ W) >= 0.5
    return float(np.mean(pred == (L_te >= 0.5)))


def _train_oracle(net, X, y, epochs, lr, batch, seed):
    """Train the fenced backprop oracle (DendriticMLP mode='oracle'). Same shuffle discipline as _train."""
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode="oracle", lr=lr)


# ============================================================================================================
# STAGE 0 -- depth-genuineness: does V1-CIFAR-k NEED depth? (the load-bearing precondition, what XOR/MNIST FAILED)
# ============================================================================================================
def stage0_depth_genuineness(task, k, hidden, epochs, lr, batch, seed):
    """A 1-hidden-layer fenced-backprop oracle must UNDERFIT held-out while a 2-3-hidden-layer oracle clears it by a
    clear margin (>= 0.80 AND >= 1-layer + a real gap). ALL arms are the fenced backprop ORACLE (this measures the
    representational depth-requirement of the TASK, not the credit rule) -- the honest 'is this the right k' gate.
    Reports 0/1/2/3 hidden-layer oracle held-out + the depth gaps + the chance floor."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    n_in = Xtr.shape[1]
    chance = float(max(np.mean(yte == c) for c in np.unique(yte)))

    def _oracle(sizes):
        net = DendriticMLP(sizes, seed=seed)
        _train_oracle(net, Xtr, ytr, epochs, lr, batch, seed)
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    l0_tr, l0_te = _oracle([n_in, k])                                      # linear (no hidden) floor
    l1_tr, l1_te = _oracle([n_in, hidden, k])                             # 1 hidden layer
    l2_tr, l2_te = _oracle([n_in, hidden, hidden, k])                     # 2 hidden layers
    l3_tr, l3_te = _oracle([n_in, hidden, hidden, hidden, k])             # 3 hidden layers (the deep regime)
    deep_best = max(l2_te, l3_te)
    depth_gap = deep_best - max(l0_te, l1_te)                             # how much depth buys over shallow/linear
    # depth-separating iff the deep oracle clears the bar AND beats the best shallow arm by a clear margin.
    depth_separating = bool(deep_best >= 0.80 and depth_gap >= 0.05 and deep_best > l1_te + 0.03)
    return {"chance": chance, "n_features": int(n_in),
            "linear_heldout": l0_te, "linear_train": l0_tr,
            "l1_heldout": l1_te, "l1_train": l1_tr,
            "l2_heldout": l2_te, "l2_train": l2_tr,
            "l3_heldout": l3_te, "l3_train": l3_tr,
            "deep_best_heldout": deep_best, "depth_gap": float(depth_gap),
            "depth_separating": depth_separating}


# ============================================================================================================
# STAGE 1 -- the deep-credit arms + per-layer alignment (the GATED metric) + anti-cheats (incl. the Trap-B defeat).
# ============================================================================================================
def _arm_alignment_and_acc(NetCls, sizes, task, epochs, lr, batch, seed, kind,
                           feedback="fixed", homeostasis=False, kp_lr=0.2, kp_decay=1e-4, beta=1.0, p0=0.30):
    """Train one deep-credit arm (mode='bdsp' = the LEARNING update), then measure per-layer credit-alignment vs the
    oracle-backprop update on the trained net + held-out accuracy + no-weight-transport. `kind` in {'fa','burstprop'}
    selects which per-layer update the alignment metric reads (must match the net's rule)."""
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = task
    net = NetCls(sizes, seed=seed, beta=beta, p0=p0, feedback=feedback, homeostasis=homeostasis,
                 kp_lr=kp_lr, kp_decay=kp_decay)
    _train(net, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    align = _per_layer_alignment(net, ab, aby, kind)                      # [layer0=deepest, ..., output]
    if feedback == "learned":
        nwt = bool(_no_weight_transport_learned(net)
                   and (_no_weight_transport_mc(net) if isinstance(net, MicrocircuitBDSPNet) else True))
    else:
        nwt = bool((_no_weight_transport_mc(net) if isinstance(net, MicrocircuitBDSPNet)
                    else _no_weight_transport(net)))
    probe = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
    return {"heldout": float(net.accuracy(Xte, yte)), "train": float(net.accuracy(Xtr, ytr)),
            "per_layer_alignment": [float(c) for c in align], "deepest_layer_alignment": float(align[0]),
            "no_weight_transport": nwt, "probe_latent": float(probe),
            "feedback": feedback, "homeostasis": bool(homeostasis)}


def _wrongsign_alignment(NetCls, sizes, task, epochs, lr, batch, seed, kind, beta=1.0, p0=0.30):
    """THE TRAP-B DEFEAT: train a WRONG-SIGN net (negate the teaching signal), then measure per-layer ALIGNMENT. On
    MNIST wrong-sign still got 0.95 ACCURACY (the readout rescues it); here the deepest-layer ALIGNMENT must be
    near-0/negative (the hidden credit is genuinely inverted). Reports both accuracy (which may be rescued) AND
    alignment (which must fail)."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    net = NetCls(sizes, seed=seed, beta=beta, p0=p0)                       # fixed feedback (a valid control baseline)
    _train(net, Xtr, ytr, "wrong_sign", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    # measure the alignment of the LEARNING-rule update on the wrong-sign-TRAINED weights (the trained rep is inverted).
    align = _per_layer_alignment(net, ab, aby, kind)
    return {"heldout": float(net.accuracy(Xte, yte)), "train": float(net.accuracy(Xtr, ytr)),
            "per_layer_alignment": [float(c) for c in align], "deepest_layer_alignment": float(align[0])}


def stage1_deep_credit(task, k, hidden, epochs, lr, batch, seed, rule="microcircuit",
                       feedback="fixed", homeostasis=False, kp_lr=0.2, kp_decay=1e-4, beta=1.0, p0=0.30):
    """The deep-credit arms on the depth-3 net [n_in, H, H, H, k] over the V1-CIFAR-k features. GATED METRIC =
    per-layer credit-alignment vs the fenced oracle. Arms: oracle (ceiling) / microcircuit or burstprop (the selected
    rule, fixed + learned-feedback) / plain-FA / 1-layer floor / wrong-sign(alignment=Trap-B defeat) / permuted /
    lesion. All share the SAME W-init/Y-init/optimizer/oracle (within-net contrast)."""
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = task
    n_in = Xtr.shape[1]
    deep = [n_in, hidden, hidden, hidden, k]                              # depth-3 (the deep regime)
    shal = [n_in, hidden, k]                                              # 1-hidden floor
    chance = float(max(np.mean(yte == c) for c in np.unique(yte)))
    Net = MicrocircuitBDSPNet if rule == "microcircuit" else BDSPNet
    _kind = "burstprop" if rule == "burstprop" else "fa"
    res = {"rule": rule, "chance": chance}

    # oracle ceiling + its per-layer alignment reference (== 1.0 by construction; sanity)
    onet = DendriticMLP(deep, seed=seed)
    _train_oracle(onet, Xtr, ytr, epochs, lr, batch, seed)
    res["oracle"] = {"heldout": float(onet.accuracy(Xte, yte)), "train": float(onet.accuracy(Xtr, ytr))}

    # the selected deep-credit rule (fixed feedback = rung-1) + the learned-apical-feedback (KP) variant
    res["test_fixed"] = _arm_alignment_and_acc(Net, deep, task, epochs, lr, batch, seed, _kind,
                                               feedback="fixed", homeostasis=False, beta=beta, p0=p0)
    res["test_learned"] = _arm_alignment_and_acc(Net, deep, task, epochs, lr, batch, seed, _kind,
                                                 feedback="learned", homeostasis=False,
                                                 kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0)
    res["test_learned_homeo"] = _arm_alignment_and_acc(Net, deep, task, epochs, lr, batch, seed, _kind,
                                                       feedback="learned", homeostasis=True,
                                                       kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0)
    # plain-FA baseline (clean-error FA, no burst/interneuron machinery) -- fixed feedback
    res["plain_fa"] = _arm_alignment_and_acc(FANet, deep, task, epochs, lr, batch, seed, "fa",
                                             feedback="fixed", homeostasis=False, beta=beta, p0=p0)
    # 1-hidden floor (memorization/no-depth) -- fixed feedback
    fnet = Net(shal, seed=seed, beta=beta, p0=p0)
    _train(fnet, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    res["single_layer"] = {"heldout": float(fnet.accuracy(Xte, yte)), "train": float(fnet.accuracy(Xtr, ytr))}

    # --- anti-cheats ---
    # (1) THE TRAP-B DEFEAT: wrong-sign must FAIL the alignment metric (accuracy may be rescued by the readout).
    res["wrong_sign"] = _wrongsign_alignment(Net, deep, task, epochs, lr, batch, seed, _kind, beta=beta, p0=p0)
    res["wrong_sign_plain_fa"] = _wrongsign_alignment(FANet, deep, task, epochs, lr, batch, seed, "fa",
                                                      beta=beta, p0=p0)
    # (2) permuted-label -> chance (generalization, not leakage)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    pnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(pnet, Xtr, yperm, "bdsp", epochs, lr, batch, seed)
    res["permuted"] = {"heldout": float(pnet.accuracy(Xte, yte)), "train": float(pnet.accuracy(Xtr, yperm))}
    # (4) apical-lesion (Y=0) -> no top-down credit -> deepest-layer alignment collapses; held-out at floor
    lnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(lnet, Xtr, ytr, "apical_lesion", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    l_align = _per_layer_alignment(lnet, ab, aby, _kind)
    res["apical_lesion"] = {"heldout": float(lnet.accuracy(Xte, yte)), "train": float(lnet.accuracy(Xtr, ytr)),
                            "per_layer_alignment": [float(c) for c in l_align], "deepest_layer_alignment": float(l_align[0])}

    # decisive within-net contrast fairness: the deep-credit Net init == the oracle DendriticMLP init (same forward W)
    b0 = Net(deep, seed=seed, beta=beta, p0=p0); f0 = DendriticMLP(deep, seed=seed)
    res["same_init_as_oracle"] = bool(all(np.allclose(a, b) for a, b in zip(b0.W, f0.W)))
    return res


def run_seed(seed, classes, k, hidden, epochs, lr, batch, rule, feedback, homeostasis,
             kp_lr, kp_decay, beta, p0, n_per_class, feature_seed, input_mode="v1"):
    task_full = make_task_gabor_cifar(seed, classes, n_per_class=n_per_class, feature_seed=feature_seed,
                                      input_mode=input_mode)
    if task_full is None:
        return None
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta = task_full
    task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    s0 = stage0_depth_genuineness(task, k, hidden, epochs, lr, batch, seed)
    s1 = stage1_deep_credit(task, k, hidden, epochs, lr, batch, seed, rule=rule, feedback=feedback,
                            homeostasis=homeostasis, kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0)
    return {"seed": seed, "meta": meta, "stage0_depth_genuineness": s0, "stage1_deep_credit": s1}


def _fmt_align(a):
    return "[" + ", ".join(f"{c:.2f}" for c in a) + "]"


def main():
    ap = argparse.ArgumentParser(description="Gabor-CIFAR-k deep-credit de-risk (rate-numpy reference).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--classes", type=int, default=3, help="k = number of CIFAR-10 classes (the first k class ids); "
                    "or use --class-ids for an explicit confusable subset.")
    ap.add_argument("--class-ids", type=int, nargs="+", default=None,
                    help="explicit CIFAR-10 class ids (overrides --classes); e.g. 3 5 4 7 = cat/dog/deer/horse.")
    ap.add_argument("--hidden", type=int, default=64, help="hidden width (CPU smoke 64; controller GPU run wider).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-per-class", type=int, default=600, help="images per class (65/35 train/held split).")
    ap.add_argument("--feature-seed", type=int, default=0,
                    help="0 = raw V1-complex features; >0 = a fixed decorrelating random projection to that many dims "
                         "(tune the depth-genuineness window; label-free).")
    ap.add_argument("--rule", choices=["burstprop", "microcircuit"], default="microcircuit")
    ap.add_argument("--feedback", choices=["fixed", "learned"], default="fixed",
                    help="TEST-arm apical feedback for the headline read (both fixed+learned are always computed).")
    ap.add_argument("--homeostasis", action="store_true")
    ap.add_argument("--kp-lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--input-mode", choices=["v1", "raw", "rawrgb"], default="v1",
                    help="v1 = fixed Gabor/V1 front end (too good => task linearly decodable, depth NOT required); "
                         "raw = raw grayscale pixels (1024-dim, NOT linearly separable => depth genuinely required); "
                         "rawrgb = raw RGB (3072-dim).")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    k = len(a.class_ids) if a.class_ids else a.classes
    classes = a.class_ids if a.class_ids else a.classes

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, classes, k, a.hidden, a.epochs, a.lr, a.batch, a.rule, a.feedback, a.homeostasis,
                         a.kp_lr, a.kp_decay, a.beta, a.p0, a.n_per_class, a.feature_seed, a.input_mode)
            if r is None:
                print("[NOT RUNNABLE] CIFAR-10 cache absent -- run: python research/findings/raw/_download_cifar.py")
                return 2
            per.append(r)
            s0 = r["stage0_depth_genuineness"]; s1 = r["stage1_deep_credit"]
            m = r["meta"]
            print("-" * 108, flush=True)
            print(f"[seed {s}] CIFAR classes {m['class_names']} | {m['n_features']} V1-complex features "
                  f"(feature_seed {m['feature_seed']}) | {m['n_train']} train / {m['n_heldout']} held | chance {s0['chance']:.3f}",
                  flush=True)
            print(f"  STAGE0 depth-genuineness: linear {s0['linear_heldout']:.3f} | 1-layer {s0['l1_heldout']:.3f} | "
                  f"2-layer {s0['l2_heldout']:.3f} | 3-layer {s0['l3_heldout']:.3f} | deep-best {s0['deep_best_heldout']:.3f} "
                  f"| depth-gap {s0['depth_gap']:+.3f} => DEPTH-SEPARATING {s0['depth_separating']}", flush=True)
            tf = s1["test_fixed"]; tl = s1["test_learned"]; th = s1["test_learned_homeo"]; pf = s1["plain_fa"]
            ws = s1["wrong_sign"]; les = s1["apical_lesion"]
            print(f"  STAGE1 [{s1['rule']}] per-layer align vs oracle (layer0=deepest):", flush=True)
            print(f"    test-fixed   held {tf['heldout']:.3f} align {_fmt_align(tf['per_layer_alignment'])} deep {tf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    test-learned held {tl['heldout']:.3f} align {_fmt_align(tl['per_layer_alignment'])} deep {tl['deepest_layer_alignment']:.2f} (KP, transport-free {tl['no_weight_transport']})", flush=True)
            print(f"    +homeo       held {th['heldout']:.3f} align {_fmt_align(th['per_layer_alignment'])} deep {th['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    plain-FA     held {pf['heldout']:.3f} align {_fmt_align(pf['per_layer_alignment'])} deep {pf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    single-layer held {s1['single_layer']['heldout']:.3f} | oracle-d3 {s1['oracle']['heldout']:.3f} | chance {s1['chance']:.3f}", flush=True)
            print(f"    [anti-cheat] WRONG-SIGN: acc {ws['heldout']:.3f} (may be rescued) | ALIGNMENT deep {ws['deepest_layer_alignment']:.2f} "
                  f"align {_fmt_align(ws['per_layer_alignment'])}  <- must FAIL (Trap-B defeat)", flush=True)
            print(f"    [anti-cheat] permuted {s1['permuted']['heldout']:.3f} (~chance) | lesion acc {les['heldout']:.3f} align deep "
                  f"{les['deepest_layer_alignment']:.2f} | same-init {s1['same_init_as_oracle']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    # ---- aggregate + pre-registered reads (descriptive; the decisive on-bridge spiking arm is the controller's GPU run) ----
    summary = {"probe": "gabor_cifar_deep_credit", "seeds": a.seeds, "rule": a.rule,
               "config": {"classes": (a.class_ids or a.classes), "k": k, "hidden": a.hidden, "epochs": a.epochs,
                          "lr": a.lr, "batch": a.batch, "n_per_class": a.n_per_class, "feature_seed": a.feature_seed,
                          "feedback": a.feedback, "homeostasis": bool(a.homeostasis), "kp_lr": a.kp_lr,
                          "kp_decay": a.kp_decay, "beta": a.beta, "p0": a.p0, "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(path):
            out = []
            for p in per:
                v = p
                for kk in path:
                    v = v[kk]
                out.append(v)
            return float(np.mean(out))
        s0_sep = all(p["stage0_depth_genuineness"]["depth_separating"] for p in per)
        deep_best = _m(["stage0_depth_genuineness", "deep_best_heldout"])
        l1 = _m(["stage0_depth_genuineness", "l1_heldout"]); depth_gap = _m(["stage0_depth_genuineness", "depth_gap"])
        oracle = _m(["stage1_deep_credit", "oracle", "heldout"])
        tf_deep = _m(["stage1_deep_credit", "test_fixed", "deepest_layer_alignment"])
        tl_deep = _m(["stage1_deep_credit", "test_learned", "deepest_layer_alignment"])
        th_deep = _m(["stage1_deep_credit", "test_learned_homeo", "deepest_layer_alignment"])
        pf_deep = _m(["stage1_deep_credit", "plain_fa", "deepest_layer_alignment"])
        best_test_deep = max(tf_deep, tl_deep, th_deep)
        ws_deep = _m(["stage1_deep_credit", "wrong_sign", "deepest_layer_alignment"])
        les_deep = _m(["stage1_deep_credit", "apical_lesion", "deepest_layer_alignment"])
        perm = _m(["stage1_deep_credit", "permuted", "heldout"]); ch = _m(["stage1_deep_credit", "chance"])
        wt = all(p["stage1_deep_credit"]["test_learned"]["no_weight_transport"]
                 and p["stage1_deep_credit"]["same_init_as_oracle"] for p in per)
        # pre-registered reads (per the scoping): Stage-0 gates the whole thing; then the alignment signal.
        wrongsign_fails = bool(ws_deep < best_test_deep - 0.10 and ws_deep < 0.30)   # Trap-B defeat
        lesion_collapses = bool(les_deep < best_test_deep - 0.10)
        permuted_chance = bool(perm <= ch + 0.08)
        align_signal = bool(best_test_deep > pf_deep - 0.02)     # the credit arm's deep alignment is at least as good as plain-FA
        oracle_ok = bool(oracle >= 0.80)
        # the load-bearing SIGNAL read (the builder reports; the controller runs the 6-seed + on-bridge GO):
        signal = bool(s0_sep and oracle_ok and best_test_deep > 0.15 and wrongsign_fails and lesion_collapses
                      and permuted_chance and wt)
        if not s0_sep:
            read = (f"STAGE-0 BOUNDARY -- V1-CIFAR-k is NOT depth-separating at k={k}/H{a.hidden}/features="
                    f"{per[0]['meta']['n_features']} (deep-best {deep_best:.3f} vs 1-layer {l1:.3f}, gap {depth_gap:+.3f}). "
                    f"This is the wrong k, NOT a GO -- escalate (more classes / fewer-decorrelated features via "
                    f"--feature-seed / harder confusable subset via --class-ids 3 5 4 7) BEFORE reading the deep-credit "
                    f"arms. This gate is what XOR/MNIST FAILED; it must PASS first.")
        elif not oracle_ok:
            read = (f"INCONCLUSIVE -- the depth-3 oracle only reached {oracle:.3f} held-out at H{a.hidden}; tune "
                    f"epochs/lr/hidden before reading the deep-credit arms (NOT a verdict).")
        else:
            _tb = "FAILS (deep align {:.2f} < best-test {:.2f})".format(ws_deep, best_test_deep) if wrongsign_fails \
                  else "does NOT fail (deep align {:.2f}) -- Trap B may re-bite; report honestly".format(ws_deep)
            read = (f"STAGE-0 PASS (depth-separating: deep-best {deep_best:.3f} vs 1-layer {l1:.3f}, gap {depth_gap:+.3f}, "
                    f"oracle {oracle:.3f}). STAGE-1 deep credit ({a.rule}): deepest-layer alignment vs oracle -- plain-FA "
                    f"{pf_deep:.2f}, test-fixed {tf_deep:.2f}, test-learned(KP) {tl_deep:.2f}, +homeo {th_deep:.2f} "
                    f"(best {best_test_deep:.2f}); lesion collapses to {les_deep:.2f}; WRONG-SIGN alignment {_tb}; "
                    f"permuted {perm:.3f} (~chance {ch:.3f}); no weight transport {wt}. "
                    f"{'LOAD-BEARING per-layer-alignment SIGNAL on real V1-CIFAR' if signal else 'NO clean deep-credit signal yet (see the arm table)'} "
                    f"=> {'controller runs 6-seed + on-bridge spiking' if signal else 'honest read: escalate / diagnose'}. "
                    f"Numpy RATE reference; the decisive on-bridge spiking depth-3 multi-seed is the controller's GPU run.")
        summary["stage0_depth_separating"] = s0_sep
        summary["aggregate"] = {"deep_best_heldout": deep_best, "l1_heldout": l1, "depth_gap": depth_gap,
                                "oracle_heldout": oracle, "deepest_align_test_fixed": tf_deep,
                                "deepest_align_test_learned": tl_deep, "deepest_align_test_learned_homeo": th_deep,
                                "deepest_align_plain_fa": pf_deep, "best_test_deep_align": best_test_deep,
                                "wrong_sign_deep_align": ws_deep, "wrong_sign_fails_alignment": wrongsign_fails,
                                "lesion_deep_align": les_deep, "lesion_collapses": lesion_collapses,
                                "permuted_heldout": perm, "permuted_chance": permuted_chance,
                                "no_weight_transport": wt, "chance": ch}
        summary["SIGNAL"] = signal
        summary["verdict"] = read
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[gabor-cifar-deep-credit] {summary['verdict']}", flush=True)
    print(f"[gabor-cifar-deep-credit] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
