"""TRANSITIVE-INFERENCE DEEP-CREDIT DE-RISK (rate-numpy reference) -- the SHARPENED goal-relevant real-task instrument
for the validated bio-plausible deep-credit rule, after the arc converged (parts 1/2/5) on the conclusion that deep
credit's depth-benefit lives in COMPOSITION / RELATIONAL REASONING over the codes, NOT in perception-decoding (CIFAR =
convolutional) or embedding-category-decoding (real-word inheritance is LINEAR: word embeddings make categories
linearly separable, Levy-Goldberg 2014). Part-2 got traction (0.69, 5/6) on a synthetic XOR-over-pool COMPOSITION;
THIS test asks the cleaner, sharper question: does supervised deep credit learn TRANSITIVE INFERENCE -- the cleanest
depth-required relational composition -- where depth is REQUIRED by the multi-hop chaining, not by an XOR encoding?

WHY TRANSITIVE INFERENCE (the cleanest depth-required composition; EMERGE-28 did it UNSUPERVISED, here SUPERVISED):
  A linear order over N entities A>B>C>D>E>F>G. TRAIN only ADJACENT pairs (A>B, B>C, C>D, ...); TEST held-out
  NON-ADJACENT pairs (B>D, A>C, C>F, ...) -- which REQUIRE composing the chain (multi-hop), not memorizing. Entity
  representations = ARBITRARY random codes (Gaussian), so the relation is genuinely LEARNED + COMPOSED, NOT derivable
  from the codes -- this is the CLEAN depth-source: unlike real word embeddings (linear categories) or the part-2 XOR
  encoding (nonlinear-but-per-item), the ordinal rank of an entity is a function ONLY of its position in the trained
  chain, so a held-out non-adjacent judgment is UN-derivable from the codes and CAN ONLY be composed. A 1-hidden-layer
  net can memorize the adjacent-pair map but CANNOT systematically compose the transitive chain for held-out
  non-adjacent pairs; a deeper net that BUILDS the ordinal/positional structure of each entity (a nonlinear map from
  arbitrary code -> rank) can compare positions and generalizes. (Dusek-Eichenbaum 1997 transitive inference, catalog
  D.02; EMERGE-28; the value-transitivity / positional-encoding literature -- the SYMBOLIC-DISTANCE effect, accuracy
  RISING with |i-j|, is the classic behavioral signature of an integrated order rather than memorized pairs.)

THE TASK CONSTRUCTION (the honest, clean depth lever):
  - N entities, each an ARBITRARY random code (code_dim Gaussian), ranked 0..N-1 (0 = greatest).
  - A pair (i, j) input X = [code_i ; code_j] (concatenation). Target y = 1 if i > j (i is GREATER, i.e. rank_i <
    rank_j), else 0. Each unordered pair is presented in BOTH orders (i,j) and (j,i) -> classes balanced, chance 0.5.
  - TRAIN split = ADJACENT pairs |rank_i - rank_j| == 1 (both orders). The whole adjacency chain.
  - TEST held-out split = NON-ADJACENT pairs |rank_i - rank_j| >= 2 (both orders) -> genuinely require composition.
  - The CRITICAL internal pairs (EMERGE-28): items in the interior appear as BOTH a greater and a lesser item across
    the adjacent premises (A>B, B>C => B is lesser then greater), so an internal non-adjacent judgment (B>D) is
    UNSOLVABLE by per-item associative strength; only integrating the premises into an ORDER answers it.

STAGE 0 (the self-correcting gate -- MEASURED FIRST): on the held-out NON-ADJACENT split, a 1-hidden-layer fenced
  backprop oracle must UNDERFIT while a 2-3-hidden-layer oracle CLEARS it by a real margin (the depth gap ON THE
  HELD-OUT non-adjacent pairs = the composition test). A LINEAR probe on the pair-codes must be NEAR CHANCE for the
  held-out non-adjacent pairs (= not linearly memorizable from arbitrary codes). If NOT depth-separating (a shallow net
  already generalizes) it is the WRONG config -- reported honestly (tune N / code_dim / noise / obs), do NOT force
  Stage 1. This gate is what XOR/MNIST/FC-CIFAR/raw-PPMI FAILED to be goal-relevant on.

STAGE 1 (once Stage 0 passes): the deep-credit arms (microcircuit / KP-learned / plain-FA / burstprop) + the 1-layer
  floor + oracle ceiling. GATED METRIC = held-out NON-ADJACENT accuracy (valid because it genuinely requires
  composition -- the 1-layer floor is the Trap-B control) AND per-layer credit-alignment vs the fenced oracle. THE
  TRANSITIVE SIGNATURE: held-out accuracy RISES with symbolic distance |i-j| (report the per-distance curve).

ANTI-CHEATS (mandatory): (1) wrong-sign FAILS alignment (Trap-B defeat); (2) permuted-label -> chance; (3) 1-layer
  floor UNDERFITS held-out non-adjacent; (4) oracle ceiling >= 0.80 held-out non-adjacent; (5) no weight transport;
  (6) MEMORIZATION control (the EMERGE-28 broken-chain): a "bridge" adjacent link is DROPPED from training, splitting
  the chain into two disconnected sub-chains -> held-out pairs that SPAN the two sub-chains have NO composing path and
  MUST be at chance (a LEAK -> them being inferable would show as memctrl accuracy > chance). (7) symbolic-distance
  effect present (the transitive signature -- reported, not a hard gate).

REUSE-BY-IMPORT (NO `sim/` edit):
  - the deep-credit ARMS + per-layer-alignment + no-weight-transport probes from `_gnw_d1_spiking_bdsp_derisk`
    (FANet / MicrocircuitBDSPNet / BDSPNet + _train + _per_layer_alignment + _no_weight_transport*).
  - `sim.dendritic_mlp.DendriticMLP` -- the fenced backprop ORACLE (Stage-0 depth oracle + Stage-1 ceiling + the
    per-layer-alignment reference) + the 1-hidden-layer floor.
  - the Stage-0/Stage-1/anti-cheat SCAFFOLD mirrored from `_semantic_inheritance_deep_credit_derisk` VERBATIM in
    structure -- only `make_task_*` differs (transitive inference vs semantic inheritance).

HONEST SCOPE: numpy RATE reference (the builder's fast CPU smoke). The on-bridge spiking depth-3 multi-seed is the
controller's decisive GPU run. NO `sim/` edit anywhere (all reuse-by-import). CPU numpy backend.

Run (1-seed smoke -- the tuned depth-separating default: n_entities=7, code_dim=16, deep-layers=2):
    SIM_BACKEND=numpy python -m research.runners._transitive_inference_deep_credit_derisk --seeds 42

The CONTROLLER's multi-seed run (fan one process per seed across cores; aggregate the per-seed JSONs):
    for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -m \
        research.runners._transitive_inference_deep_credit_derisk --seeds $s \
        --out research/findings/raw/_transitive_inference_seed$s.json & done; wait
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
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
# --- reuse-by-import: the deep-credit arms + per-layer alignment + no-weight-transport probes ---
from research.runners._gnw_d1_spiking_bdsp_derisk import (  # noqa: E402
    BDSPNet, MicrocircuitBDSPNet, FANet, _train, _per_layer_alignment,
    _no_weight_transport, _no_weight_transport_learned, _no_weight_transport_mc)

OUT = _REPO / "research" / "findings" / "raw" / "_transitive_inference_deep_credit.json"


# ============================================================================================================
# The transitive-inference composition task.
#   N entities in a linear order (rank 0 = greatest .. N-1 = least); each = an ARBITRARY random code (Gaussian).
#   Pair (i, j) -> X = [code_i ; code_j]; y = 1 iff i is GREATER than j (rank_i < rank_j), else 0. Both orders.
#   TRAIN = ADJACENT pairs (|rank diff| == 1); TEST held-out = NON-ADJACENT pairs (|rank diff| >= 2) -> composition.
#   MEMORIZATION control (broken-chain): a "bridge" adjacent link is dropped from training so the chain splits into two
#     disconnected sub-chains; held-out pairs that SPAN the two sub-chains have no composing path -> must be at chance.
# ============================================================================================================
def make_task_transitive_inference(seed, n_entities=7, code_dim=16, n_obs=24, noise=0.10, bridge_gap=None,
                                    feature_seed=0):
    """Build the transitive-inference deep-credit task.

    Returns (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx -- the SAME shape the _gnw_d1 arms consume.
      - y = 1 iff the FIRST entity of the pair is GREATER (rank smaller). ADJACENT pairs are the training targets;
        NON-ADJACENT pairs are held out (in Xte only) -> generalizing to them = the composition (multi-hop) test.
      - Ltr/Lte latents (secondary/reported emergence probe) = a one-hot-ish ordinal RANK code of the FIRST entity
        (the intermediate positional feature the composition must build). NOT the gate.
      - idx = {inh_idx: held-out non-adjacent pairs WITHIN one connected sub-chain (the genuine composition test);
               memctrl_idx: held-out pairs SPANNING the two broken-chain sub-chains (must NOT be inferable);
               dist_of_row: |rank_i - rank_j| per held-out row (for the symbolic-distance curve)}.

    THE DEPTH LEVER (why this is genuinely depth-required): the entity codes are ARBITRARY random Gaussians, so an
    entity's RANK is NOT a linear function of its code -- recovering rank(code) is a nonlinear map (needs a hidden
    layer), and COMPARING two recovered ranks to decide "greater" for a held-out NON-ADJACENT pair needs the ranks to
    be composed on a common ordinal scale that was only ever taught via ADJACENT links (multi-hop chaining = a second
    composition). A 1-layer / linear net memorizes the adjacent-pair -> label map (each trained pair is individually
    addressable) but has no ordinal scale to place a never-seen non-adjacent pair on -> it UNDERFITS held-out
    non-adjacent; a 2-3-layer net that BUILDS the ordinal embedding of each code generalizes the transitive order. The
    SYMBOLIC-DISTANCE effect (held-out accuracy rising with |rank_i - rank_j|) is the behavioral signature of the
    integrated order (Dusek-Eichenbaum 1997; catalog D.02)."""
    rng = np.random.default_rng(seed)
    n = int(n_entities)
    # ARBITRARY random entity codes (Gaussian, unit-ish). The linear order is over the ENTITY INDICES 0..n-1
    # (rank 0 = greatest); the codes carry NO ordinal information (a random permutation of codes-to-ranks would be a
    # different task with the same statistics) -> the relation is genuinely learned, not read off the code.
    crng = np.random.default_rng(seed * 131 + 7)
    codes = crng.standard_normal((n, code_dim)) / np.sqrt(code_dim)  # (n, code_dim) arbitrary codes

    # MEMORIZATION control (the EMERGE-28 broken-chain): DROP one adjacent "bridge" link from TRAINING so the chain
    # splits into two disconnected sub-chains {0..gap} and {gap+1..n-1}. Held-out pairs SPANNING the two sub-chains have
    # NO composing path through the trained premises -> a faithful net MUST be at chance on them (they are genuinely not
    # inferable). A LEAK (the net inferring a spanning pair) shows as memctrl accuracy > chance. bridge_gap = the rank
    # of the upper endpoint of the dropped link (dropped link = (bridge_gap, bridge_gap+1)); default = the middle.
    if bridge_gap is None:
        bridge_gap = (n // 2) - 1                                    # dropped adjacent link = (bridge_gap, bridge_gap+1)
    bridge_gap = int(bridge_gap)
    dropped = (bridge_gap, bridge_gap + 1)

    def _side(rk):
        return 0 if rk <= bridge_gap else 1                          # which broken-chain sub-chain a rank belongs to

    def _obs(idx, k):
        """One noisy observation of entity `idx`'s code (denoising is part of the recovery work)."""
        return codes[idx] + noise * k.standard_normal(code_dim)

    Xtr_l, ytr_l, Ltr_l = [], [], []
    Xte_l, yte_l, Lte_l = [], [], []
    inh_rows, mem_rows, dist_rows = [], [], []                       # held-out row bookkeeping

    orng = np.random.default_rng(seed * 977 + 3)                     # observation-noise stream
    rank_dim = n                                                     # ordinal rank code dim (emergence-probe target)

    def _rank_latent(i):
        v = np.zeros(rank_dim, dtype=np.float64)
        v[i] = 1.0                                                   # one-hot rank of the FIRST entity (positional feat)
        return v

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = abs(i - j)
            y = 1 if i < j else 0                                    # rank_i < rank_j => i is GREATER => label 1
            is_adjacent = (d == 1)
            # is this pair's adjacency link the DROPPED bridge link? (only relevant to the adjacent training set)
            link = (min(i, j), max(i, j))
            train_this = is_adjacent and (link != dropped)
            if train_this:
                for _ in range(n_obs):
                    x = np.concatenate([_obs(i, orng), _obs(j, orng)])
                    Xtr_l.append(x); ytr_l.append(y); Ltr_l.append(_rank_latent(i))
            elif not is_adjacent:
                # held-out NON-ADJACENT pair. ONE noisy observation (the held-out members appear but are never trained).
                x = np.concatenate([_obs(i, orng), _obs(j, orng)])
                row = len(Xte_l)
                Xte_l.append(x); yte_l.append(y); Lte_l.append(_rank_latent(i))
                if _side(i) == _side(j):
                    inh_rows.append(row)                            # within a connected sub-chain = genuine composition
                else:
                    mem_rows.append(row)                           # spans the broken chain = must NOT be inferable
                dist_rows.append((row, d))
            # the dropped adjacent link itself: NOT trained AND not a non-adjacent held-out pair -> omit entirely
            # (its rows would be a trivial d==1 memorization test, not the composition question).

    Xtr = np.asarray(Xtr_l); ytr = np.asarray(ytr_l, np.int64); Ltr = np.asarray(Ltr_l)
    Xte = np.asarray(Xte_l); yte = np.asarray(yte_l, np.int64); Lte = np.asarray(Lte_l)

    # optional fixed decorrelating random projection (feature_seed>0): rotate/compress the feature space (label-free)
    # so the depth-genuineness window can be tuned without leakage. feature_seed=0 => identity.
    if feature_seed and feature_seed > 0:
        frng = np.random.default_rng(feature_seed + 100003)
        n_proj = min(Xtr.shape[1], max(16, int(feature_seed)))
        P = frng.standard_normal((Xtr.shape[1], n_proj)) / np.sqrt(Xtr.shape[1])
        Xtr = Xtr @ P; Xte = Xte @ P

    # per-feature standardization on TRAIN statistics (sigmoid MLP calibration; applied to ALL arms identically).
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True)
    Xtr = (Xtr - mu) / (sd + 1e-6); Xte = (Xte - mu) / (sd + 1e-6)

    inh_idx = np.asarray(inh_rows, dtype=np.int64)
    memctrl_idx = np.asarray(mem_rows, dtype=np.int64)
    dist_of_row = {int(r): int(d) for (r, d) in dist_rows}

    # shuffle train (deterministic per seed) so batches mix classes.
    ptr = rng.permutation(len(ytr)); Xtr, ytr, Ltr = Xtr[ptr], ytr[ptr], Ltr[ptr]

    meta = {"n_entities": n, "code_dim": code_dim, "n_obs": n_obs, "noise": noise,
            "bridge_gap": bridge_gap, "dropped_link": list(dropped), "feature_seed": int(feature_seed),
            "n_features": int(Xtr.shape[1]), "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
            "n_inherit_heldout": int(len(inh_idx)), "n_memctrl_heldout": int(len(memctrl_idx)),
            "rank_dim": int(rank_dim)}
    return (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, {"inh_idx": inh_idx, "memctrl_idx": memctrl_idx,
                                                     "dist_of_row": dist_of_row}


# ============================================================================================================
# Emergence probe (secondary/reported): does a linear read-out of the FROZEN hidden rep recover the FIRST entity's
# ordinal RANK? (== the intermediate positional feature the composition must build.)
# ============================================================================================================
def _hidden_rep(net, X):
    acts, _lg = net._forward(np.asarray(X, float))
    return np.asarray(acts[-1])


def _probe_latents(H_tr, L_tr, H_te, L_te):
    if len(H_tr) == 0 or len(H_te) == 0:
        return float("nan")
    Xtr = np.concatenate([H_tr, np.ones((len(H_tr), 1))], 1)
    Xte = np.concatenate([H_te, np.ones((len(H_te), 1))], 1)
    lam = 1e-2 * np.eye(Xtr.shape[1]); lam[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ L_tr)
    pred = np.argmax(Xte @ W, 1)                                    # argmax over the ordinal-rank one-hot
    return float(np.mean(pred == np.argmax(L_te, 1)))


def _linear_probe_heldout(Xtr, ytr, Xte, yte, idx):
    """A LINEAR (ridge-logistic-ish least-squares) probe on the RAW pair-codes: is the held-out non-adjacent label
    linearly memorizable? Must be NEAR CHANCE (0.5) if the task is genuinely composition-required (arbitrary codes)."""
    if idx is None or len(idx) == 0:
        return float("nan")
    Atr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    lam = 1e-2 * np.eye(Atr.shape[1]); lam[-1, -1] = 0.0
    t = (ytr.astype(np.float64) * 2.0 - 1.0)                        # +-1 target
    W = np.linalg.solve(Atr.T @ Atr + lam, Atr.T @ t)
    Ate = np.concatenate([Xte[idx], np.ones((len(idx), 1))], 1)
    pred = (Ate @ W) >= 0.0
    return float(np.mean(pred == (yte[idx] >= 0.5)))


def _train_oracle(net, X, y, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode="oracle", lr=lr)


def _acc_on(net, X, y, idx):
    """Held-out accuracy on a SUBSET of the held set (the composition test), not the whole held set."""
    if idx is None or len(idx) == 0:
        return float("nan")
    _, lg = net._forward(np.asarray(X[idx], float))
    return float(np.mean(np.argmax(np.asarray(lg), 1) == np.asarray(y[idx])))


def _acc_by_distance(net, X, y, dist_of_row, inh_idx):
    """The SYMBOLIC-DISTANCE curve (the transitive signature): held-out accuracy on the composition subset (inh_idx)
    bucketed by |rank_i - rank_j|. Returns {distance: accuracy} + the Spearman-ish monotone slope (accuracy should
    RISE with distance)."""
    if inh_idx is None or len(inh_idx) == 0:
        return {}, float("nan")
    _, lg = net._forward(np.asarray(X[inh_idx], float))
    pred = np.argmax(np.asarray(lg), 1)
    yv = np.asarray(y[inh_idx])
    dv = np.array([dist_of_row.get(int(r), -1) for r in inh_idx])
    curve = {}
    for d in sorted(set(dv.tolist())):
        m = dv == d
        if m.sum() > 0:
            curve[int(d)] = float(np.mean(pred[m] == yv[m]))
    # slope of accuracy vs distance (positive = the transitive symbolic-distance effect).
    if len(curve) >= 2:
        ds = np.array(sorted(curve.keys()), float); accs = np.array([curve[int(d)] for d in ds])
        slope = float(np.polyfit(ds, accs, 1)[0])
    else:
        slope = float("nan")
    return curve, slope


# ============================================================================================================
# STAGE 0 -- depth-genuineness on the HELD-OUT NON-ADJACENT split (the load-bearing gate, MEASURED FIRST).
# ============================================================================================================
def stage0_depth_genuineness(task, idx, k, hidden, epochs, lr, batch, seed):
    """A 1-hidden-layer fenced-backprop oracle must UNDERFIT held-out NON-ADJACENT inference while a 2-3-hidden-layer
    oracle clears it by a clear margin. ALL arms are the fenced backprop ORACLE (this measures the representational
    depth-requirement of the TASK, not the credit rule). The gated accuracy is on the composition subset (inh_idx =
    held-out non-adjacent pairs WITHIN a connected sub-chain). Reports 0/1/2/3-hidden oracle + gaps + chance + the
    LINEAR probe on the raw pair-codes (must be near chance = not linearly memorizable)."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    inh_idx = idx["inh_idx"]
    n_in = Xtr.shape[1]
    if len(inh_idx):
        yv = yte[inh_idx]
        chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    lin_probe = _linear_probe_heldout(Xtr, ytr, Xte, yte, inh_idx)

    def _oracle(sizes):
        net = DendriticMLP(sizes, seed=seed)
        _train_oracle(net, Xtr, ytr, epochs, lr, batch, seed)
        return float(net.accuracy(Xtr, ytr)), _acc_on(net, Xte, yte, inh_idx)

    l0_tr, l0_te = _oracle([n_in, k])                                      # linear (no hidden) floor
    l1_tr, l1_te = _oracle([n_in, hidden, k])                             # 1 hidden layer
    l2_tr, l2_te = _oracle([n_in, hidden, hidden, k])                     # 2 hidden layers
    l3_tr, l3_te = _oracle([n_in, hidden, hidden, hidden, k])             # 3 hidden layers (the deep regime)
    deep_best = max(l2_te, l3_te)
    depth_gap = deep_best - max(l0_te, l1_te)
    depth_separating = bool(deep_best >= 0.80 and depth_gap >= 0.05 and deep_best > l1_te + 0.03)
    return {"chance": chance, "n_features": int(n_in), "linear_probe_heldout": lin_probe,
            "linear_inherit_heldout": l0_te, "linear_train": l0_tr,
            "l1_inherit_heldout": l1_te, "l1_train": l1_tr,
            "l2_inherit_heldout": l2_te, "l2_train": l2_tr,
            "l3_inherit_heldout": l3_te, "l3_train": l3_tr,
            "deep_best_inherit_heldout": deep_best, "depth_gap": float(depth_gap),
            "depth_separating": depth_separating}


# ============================================================================================================
# STAGE 1 -- the deep-credit arms + per-layer alignment (the GATED metric) + anti-cheats (incl. the Trap-B defeat,
# the MEMORIZATION broken-chain control, and the symbolic-distance signature).
# ============================================================================================================
def _arm_alignment_and_acc(NetCls, sizes, task, idx, epochs, lr, batch, seed, kind,
                           feedback="fixed", homeostasis=False, kp_lr=0.2, kp_decay=1e-4, beta=1.0, p0=0.30):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = task
    inh_idx, mem_idx = idx["inh_idx"], idx["memctrl_idx"]
    net = NetCls(sizes, seed=seed, beta=beta, p0=p0, feedback=feedback, homeostasis=homeostasis,
                 kp_lr=kp_lr, kp_decay=kp_decay)
    _train(net, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    align = _per_layer_alignment(net, ab, aby, kind)
    if feedback == "learned":
        nwt = bool(_no_weight_transport_learned(net)
                   and (_no_weight_transport_mc(net) if isinstance(net, MicrocircuitBDSPNet) else True))
    else:
        nwt = bool((_no_weight_transport_mc(net) if isinstance(net, MicrocircuitBDSPNet)
                    else _no_weight_transport(net)))
    if len(inh_idx):
        probe = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte[inh_idx]), Lte[inh_idx])
    else:
        probe = float("nan")
    curve, slope = _acc_by_distance(net, Xte, yte, idx["dist_of_row"], inh_idx)
    return {"inherit_heldout": _acc_on(net, Xte, yte, inh_idx), "memctrl_heldout": _acc_on(net, Xte, yte, mem_idx),
            "train": float(net.accuracy(Xtr, ytr)),
            "per_layer_alignment": [float(c) for c in align], "deepest_layer_alignment": float(align[0]),
            "no_weight_transport": nwt, "probe_latent": float(probe),
            "distance_curve": curve, "distance_slope": float(slope),
            "feedback": feedback, "homeostasis": bool(homeostasis)}


def _wrongsign_alignment(NetCls, sizes, task, idx, epochs, lr, batch, seed, kind, beta=1.0, p0=0.30):
    """THE TRAP-B DEFEAT: train a WRONG-SIGN net, then measure per-layer ALIGNMENT (the deepest-layer alignment must
    be near-0/negative even if accuracy is rescued)."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    net = NetCls(sizes, seed=seed, beta=beta, p0=p0)
    _train(net, Xtr, ytr, "wrong_sign", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    align = _per_layer_alignment(net, ab, aby, kind)
    return {"inherit_heldout": _acc_on(net, Xte, yte, idx["inh_idx"]),
            "train": float(net.accuracy(Xtr, ytr)),
            "per_layer_alignment": [float(c) for c in align], "deepest_layer_alignment": float(align[0])}


def stage1_deep_credit(task, idx, k, hidden, epochs, lr, batch, seed, rule="microcircuit",
                       feedback="fixed", homeostasis=False, kp_lr=0.2, kp_decay=1e-4, beta=1.0, p0=0.30,
                       deep_layers=2):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = task
    inh_idx, mem_idx = idx["inh_idx"], idx["memctrl_idx"]
    n_in = Xtr.shape[1]
    deep = [n_in] + [hidden] * int(deep_layers) + [k]
    shal = [n_in, hidden, k]
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    Net = MicrocircuitBDSPNet if rule == "microcircuit" else BDSPNet
    _kind = "burstprop" if rule == "burstprop" else "fa"
    res = {"rule": rule, "chance": chance}

    onet = DendriticMLP(deep, seed=seed)
    _train_oracle(onet, Xtr, ytr, epochs, lr, batch, seed)
    ocurve, oslope = _acc_by_distance(onet, Xte, yte, idx["dist_of_row"], inh_idx)
    res["oracle"] = {"inherit_heldout": _acc_on(onet, Xte, yte, inh_idx),
                     "memctrl_heldout": _acc_on(onet, Xte, yte, mem_idx), "train": float(onet.accuracy(Xtr, ytr)),
                     "distance_curve": ocurve, "distance_slope": float(oslope)}

    res["test_fixed"] = _arm_alignment_and_acc(Net, deep, task, idx, epochs, lr, batch, seed, _kind,
                                               feedback="fixed", homeostasis=False, beta=beta, p0=p0)
    res["test_learned"] = _arm_alignment_and_acc(Net, deep, task, idx, epochs, lr, batch, seed, _kind,
                                                 feedback="learned", homeostasis=False,
                                                 kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0)
    res["test_learned_homeo"] = _arm_alignment_and_acc(Net, deep, task, idx, epochs, lr, batch, seed, _kind,
                                                       feedback="learned", homeostasis=True,
                                                       kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0)
    res["plain_fa"] = _arm_alignment_and_acc(FANet, deep, task, idx, epochs, lr, batch, seed, "fa",
                                             feedback="fixed", homeostasis=False, beta=beta, p0=p0)
    # 1-hidden floor (memorization/no-depth): must UNDERFIT held-out non-adjacent inference.
    fnet = Net(shal, seed=seed, beta=beta, p0=p0)
    _train(fnet, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    fcurve, fslope = _acc_by_distance(fnet, Xte, yte, idx["dist_of_row"], inh_idx)
    res["single_layer"] = {"inherit_heldout": _acc_on(fnet, Xte, yte, inh_idx),
                           "memctrl_heldout": _acc_on(fnet, Xte, yte, mem_idx), "train": float(fnet.accuracy(Xtr, ytr)),
                           "distance_curve": fcurve, "distance_slope": float(fslope)}

    # --- anti-cheats ---
    res["wrong_sign"] = _wrongsign_alignment(Net, deep, task, idx, epochs, lr, batch, seed, _kind, beta=beta, p0=p0)
    res["wrong_sign_plain_fa"] = _wrongsign_alignment(FANet, deep, task, idx, epochs, lr, batch, seed, "fa",
                                                      beta=beta, p0=p0)
    # permuted-label -> chance on non-adjacent held-out (generalization, not leakage)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    pnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(pnet, Xtr, yperm, "bdsp", epochs, lr, batch, seed)
    res["permuted"] = {"inherit_heldout": _acc_on(pnet, Xte, yte, inh_idx), "train": float(pnet.accuracy(Xtr, yperm))}
    # apical-lesion (Y=0) -> no top-down credit -> deepest-layer alignment collapses; non-adjacent held-out at floor.
    lnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(lnet, Xtr, ytr, "apical_lesion", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    l_align = _per_layer_alignment(lnet, ab, aby, _kind)
    res["apical_lesion"] = {"inherit_heldout": _acc_on(lnet, Xte, yte, inh_idx), "train": float(lnet.accuracy(Xtr, ytr)),
                            "per_layer_alignment": [float(c) for c in l_align],
                            "deepest_layer_alignment": float(l_align[0])}

    b0 = Net(deep, seed=seed, beta=beta, p0=p0); f0 = DendriticMLP(deep, seed=seed)
    res["same_init_as_oracle"] = bool(all(np.allclose(a, b) for a, b in zip(b0.W, f0.W)))
    return res


def run_seed(seed, k, hidden, epochs, lr, batch, rule, feedback, homeostasis, kp_lr, kp_decay, beta, p0,
             task_kwargs, deep_layers=2):
    task_full = make_task_transitive_inference(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    if k is None:
        k = 2                                                       # binary greater/lesser
    s0 = stage0_depth_genuineness(task, idx, k, hidden, epochs, lr, batch, seed)
    s1 = stage1_deep_credit(task, idx, k, hidden, epochs, lr, batch, seed, rule=rule, feedback=feedback,
                            homeostasis=homeostasis, kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0,
                            deep_layers=deep_layers)
    return {"seed": seed, "meta": meta, "stage0_depth_genuineness": s0, "stage1_deep_credit": s1}


def _fmt_align(a):
    return "[" + ", ".join(f"{c:.2f}" for c in a) + "]"


def _fmt_curve(c):
    return "{" + ", ".join(f"d{d}:{c[d]:.2f}" for d in sorted(c)) + "}"


def main():
    ap = argparse.ArgumentParser(description="Transitive-inference relational-composition deep-credit de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rule", choices=["burstprop", "microcircuit"], default="microcircuit")
    ap.add_argument("--feedback", choices=["fixed", "learned"], default="fixed")
    ap.add_argument("--homeostasis", action="store_true")
    ap.add_argument("--kp-lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    # --- task knobs (tune the depth-genuineness window) ---
    ap.add_argument("--deep-layers", type=int, default=2,
                    help="hidden layers in the DEEP arm (default 2 = where the transitive-chain composition needs depth "
                         "AND the oracle clears; 3 may overfit at this budget).")
    ap.add_argument("--n-entities", type=int, default=7, help="entities in the linear order (>= 6 for interior pairs)")
    ap.add_argument("--code-dim", type=int, default=16, help="arbitrary-random entity code dim (per entity)")
    ap.add_argument("--n-obs", type=int, default=24, help="noisy observations per trained adjacent pair")
    ap.add_argument("--noise", type=float, default=0.10, help="observation noise on the entity codes")
    ap.add_argument("--bridge-gap", type=int, default=None,
                    help="rank of the upper endpoint of the DROPPED adjacent bridge link (broken-chain memctrl); "
                         "default = middle.")
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_entities=a.n_entities, code_dim=a.code_dim, n_obs=a.n_obs, noise=a.noise,
                       bridge_gap=a.bridge_gap, feature_seed=a.feature_seed)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, None, a.hidden, a.epochs, a.lr, a.batch, a.rule, a.feedback, a.homeostasis,
                         a.kp_lr, a.kp_decay, a.beta, a.p0, task_kwargs, deep_layers=a.deep_layers)
            per.append(r)
            s0 = r["stage0_depth_genuineness"]; s1 = r["stage1_deep_credit"]; m = r["meta"]
            print("-" * 112, flush=True)
            print(f"[seed {s}] {m['n_entities']} entities (linear order) | code_dim {m['code_dim']} | "
                  f"dropped bridge link {m['dropped_link']} | {m['n_features']} feats | {m['n_train']} train / "
                  f"{m['n_inherit_heldout']} nonadj-composition-held / {m['n_memctrl_heldout']} broken-chain-held | "
                  f"chance {s0['chance']:.3f}", flush=True)
            print(f"  STAGE0 depth-genuineness (held-out NON-ADJACENT acc): linear {s0['linear_inherit_heldout']:.3f} | "
                  f"1-layer {s0['l1_inherit_heldout']:.3f} | 2-layer {s0['l2_inherit_heldout']:.3f} | "
                  f"3-layer {s0['l3_inherit_heldout']:.3f} | deep-best {s0['deep_best_inherit_heldout']:.3f} | "
                  f"depth-gap {s0['depth_gap']:+.3f} | LINEAR-probe {s0['linear_probe_heldout']:.3f} "
                  f"=> DEPTH-SEPARATING {s0['depth_separating']}", flush=True)
            tf = s1["test_fixed"]; tl = s1["test_learned"]; th = s1["test_learned_homeo"]; pf = s1["plain_fa"]
            ws = s1["wrong_sign"]; les = s1["apical_lesion"]
            print(f"  STAGE1 [{s1['rule']}] per-layer align vs oracle (layer0=deepest) + held-out NON-ADJACENT acc:",
                  flush=True)
            print(f"    test-fixed   inherit {tf['inherit_heldout']:.3f} memctrl {tf['memctrl_heldout']:.3f} "
                  f"align {_fmt_align(tf['per_layer_alignment'])} deep {tf['deepest_layer_alignment']:.2f} | "
                  f"dist {_fmt_curve(tf['distance_curve'])} slope {tf['distance_slope']:+.3f}", flush=True)
            print(f"    test-learned inherit {tl['inherit_heldout']:.3f} align {_fmt_align(tl['per_layer_alignment'])} "
                  f"deep {tl['deepest_layer_alignment']:.2f} (KP, transport-free {tl['no_weight_transport']})", flush=True)
            print(f"    +homeo       inherit {th['inherit_heldout']:.3f} align {_fmt_align(th['per_layer_alignment'])} "
                  f"deep {th['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    plain-FA     inherit {pf['inherit_heldout']:.3f} align {_fmt_align(pf['per_layer_alignment'])} "
                  f"deep {pf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    single-layer inherit {s1['single_layer']['inherit_heldout']:.3f} "
                  f"dist {_fmt_curve(s1['single_layer']['distance_curve'])} | "
                  f"oracle inherit {s1['oracle']['inherit_heldout']:.3f} dist {_fmt_curve(s1['oracle']['distance_curve'])} "
                  f"| chance {s1['chance']:.3f}", flush=True)
            print(f"    [anti-cheat] WRONG-SIGN: inherit {ws['inherit_heldout']:.3f} (may be rescued) | ALIGNMENT deep "
                  f"{ws['deepest_layer_alignment']:.2f} align {_fmt_align(ws['per_layer_alignment'])}  <- must FAIL", flush=True)
            print(f"    [anti-cheat] permuted {s1['permuted']['inherit_heldout']:.3f} (~chance) | lesion inherit "
                  f"{les['inherit_heldout']:.3f} align deep {les['deepest_layer_alignment']:.2f} | "
                  f"BROKEN-CHAIN memctrl(oracle) {s1['oracle']['memctrl_heldout']:.3f} (must ~chance) | "
                  f"same-init {s1['same_init_as_oracle']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "transitive_inference_deep_credit", "seeds": a.seeds, "rule": a.rule,
               "config": {"hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                          "feedback": a.feedback, "homeostasis": bool(a.homeostasis), "kp_lr": a.kp_lr,
                          "kp_decay": a.kp_decay, "beta": a.beta, "p0": a.p0, "task": task_kwargs,
                          "deep_layers": a.deep_layers, "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(path):
            out = []
            for p in per:
                v = p
                for kk in path:
                    v = v[kk]
                out.append(v)
            return float(np.nanmean(out))
        s0_sep = all(p["stage0_depth_genuineness"]["depth_separating"] for p in per)
        deep_best = _m(["stage0_depth_genuineness", "deep_best_inherit_heldout"])
        l1 = _m(["stage0_depth_genuineness", "l1_inherit_heldout"]); depth_gap = _m(["stage0_depth_genuineness", "depth_gap"])
        lin_probe = _m(["stage0_depth_genuineness", "linear_probe_heldout"])
        oracle = _m(["stage1_deep_credit", "oracle", "inherit_heldout"])
        oracle_mem = _m(["stage1_deep_credit", "oracle", "memctrl_heldout"])
        oracle_slope = _m(["stage1_deep_credit", "oracle", "distance_slope"])
        tf_deep = _m(["stage1_deep_credit", "test_fixed", "deepest_layer_alignment"])
        tl_deep = _m(["stage1_deep_credit", "test_learned", "deepest_layer_alignment"])
        th_deep = _m(["stage1_deep_credit", "test_learned_homeo", "deepest_layer_alignment"])
        pf_deep = _m(["stage1_deep_credit", "plain_fa", "deepest_layer_alignment"])
        best_test_deep = max(tf_deep, tl_deep, th_deep)
        tf_inh = _m(["stage1_deep_credit", "test_fixed", "inherit_heldout"])
        tl_inh = _m(["stage1_deep_credit", "test_learned", "inherit_heldout"])
        th_inh = _m(["stage1_deep_credit", "test_learned_homeo", "inherit_heldout"])
        best_test_inh = max(tf_inh, tl_inh, th_inh)
        tf_slope = _m(["stage1_deep_credit", "test_fixed", "distance_slope"])
        sl_inh = _m(["stage1_deep_credit", "single_layer", "inherit_heldout"])
        ws_deep = _m(["stage1_deep_credit", "wrong_sign", "deepest_layer_alignment"])
        les_deep = _m(["stage1_deep_credit", "apical_lesion", "deepest_layer_alignment"])
        les_inh = _m(["stage1_deep_credit", "apical_lesion", "inherit_heldout"])
        perm = _m(["stage1_deep_credit", "permuted", "inherit_heldout"]); ch = _m(["stage1_deep_credit", "chance"])
        wt = all(p["stage1_deep_credit"]["test_learned"]["no_weight_transport"]
                 and p["stage1_deep_credit"]["same_init_as_oracle"] for p in per)
        wrongsign_fails = bool(ws_deep < best_test_deep - 0.10 and ws_deep < 0.30)
        lesion_collapses = bool(les_deep < best_test_deep - 0.10)
        permuted_chance = bool(perm <= ch + 0.08)
        align_signal = bool(best_test_deep > pf_deep - 0.02)
        # the goal-relevant read: the deep-credit rule LEARNS the composition (held-out non-adjacent > 1-layer floor)
        learns_composition = bool(best_test_inh > sl_inh + 0.05 and best_test_inh > ch + 0.05)
        memctrl_holds = bool(np.isnan(oracle_mem) or oracle_mem <= ch + 0.15)   # the broken-chain no-leakage control
        symbolic_distance = bool(tf_slope > 0.0)                                # the transitive signature (reported)
        oracle_ok = bool(oracle >= 0.80)
        signal = bool(s0_sep and oracle_ok and best_test_deep > 0.15 and learns_composition and wrongsign_fails
                      and lesion_collapses and permuted_chance and wt and memctrl_holds)
        if not s0_sep:
            read = (f"STAGE-0 BOUNDARY -- the transitive-inference task is NOT depth-separating at n_entities="
                    f"{a.n_entities}/code_dim={a.code_dim}/noise={a.noise}/n_obs={a.n_obs} (deep-best non-adjacent "
                    f"{deep_best:.3f} vs 1-layer {l1:.3f}, gap {depth_gap:+.3f}, linear-probe {lin_probe:.3f}). This is "
                    f"the wrong task CONFIG, NOT a GO -- escalate (more entities via --n-entities so more interior "
                    f"non-adjacent pairs / larger --code-dim so the codes are less linearly separable / more --n-obs / "
                    f"lower --noise so the depth-2 oracle reliably clears >=0.80) BEFORE reading the deep-credit arms. "
                    f"This gate is what XOR/MNIST/FC-CIFAR/raw-PPMI FAILED to be goal-relevant on; it must PASS first.")
        elif not oracle_ok:
            read = (f"INCONCLUSIVE -- the deep oracle only reached {oracle:.3f} held-out non-adjacent at H{a.hidden}; "
                    f"tune epochs/lr/hidden before reading the deep-credit arms (NOT a verdict).")
        else:
            _tb = "FAILS (deep align {:.2f} < best-test {:.2f})".format(ws_deep, best_test_deep) if wrongsign_fails \
                  else "does NOT fail (deep align {:.2f}) -- Trap B may re-bite; report honestly".format(ws_deep)
            _comp = "LEARNS the transitive composition" if learns_composition else "does NOT beat the 1-layer floor"
            read = (f"STAGE-0 PASS (depth-separating: deep-best non-adjacent {deep_best:.3f} vs 1-layer {l1:.3f}, gap "
                    f"{depth_gap:+.3f}, linear-probe {lin_probe:.3f} ~chance, oracle {oracle:.3f}). STAGE-1 deep credit "
                    f"({a.rule}): held-out NON-ADJACENT -- single-layer {sl_inh:.3f}, best-test {best_test_inh:.3f}, "
                    f"chance {ch:.3f} ({_comp}); symbolic-distance slope test-fixed {tf_slope:+.3f} / oracle {oracle_slope:+.3f} "
                    f"({'PRESENT' if symbolic_distance else 'absent'}); deepest-layer alignment -- plain-FA {pf_deep:.2f}, "
                    f"best-test {best_test_deep:.2f}; lesion inherit {les_inh:.3f} align {les_deep:.2f}; WRONG-SIGN "
                    f"alignment {_tb}; permuted {perm:.3f} (~chance {ch:.3f}); broken-chain memctrl(oracle) {oracle_mem:.3f} "
                    f"({'holds' if memctrl_holds else 'LEAKS'}); no weight transport {wt}. "
                    f"{'LOAD-BEARING depth-benefit on TRANSITIVE INFERENCE (the cleanest relational composition)' if signal else 'NO clean deep-credit signal yet (see the arm table)'} "
                    f"=> {'controller runs 6-seed + adversarial-verify + on-bridge spiking' if signal else 'honest read: escalate / diagnose'}. "
                    f"Numpy RATE reference; the decisive on-bridge spiking depth-3 multi-seed is the controller's GPU run.")
        summary["stage0_depth_separating"] = s0_sep
        summary["aggregate"] = {"deep_best_inherit_heldout": deep_best, "l1_inherit_heldout": l1, "depth_gap": depth_gap,
                                "linear_probe_heldout": lin_probe, "oracle_inherit_heldout": oracle,
                                "oracle_memctrl_heldout": oracle_mem, "oracle_distance_slope": oracle_slope,
                                "deepest_align_test_fixed": tf_deep, "deepest_align_test_learned": tl_deep,
                                "deepest_align_test_learned_homeo": th_deep, "deepest_align_plain_fa": pf_deep,
                                "best_test_deep_align": best_test_deep, "inherit_test_fixed": tf_inh,
                                "inherit_test_learned": tl_inh, "inherit_test_learned_homeo": th_inh,
                                "best_test_inherit": best_test_inh, "single_layer_inherit": sl_inh,
                                "test_fixed_distance_slope": tf_slope, "symbolic_distance_effect": symbolic_distance,
                                "learns_composition": learns_composition, "wrong_sign_deep_align": ws_deep,
                                "wrong_sign_fails_alignment": wrongsign_fails, "lesion_deep_align": les_deep,
                                "lesion_collapses": lesion_collapses, "permuted_inherit": perm,
                                "permuted_chance": permuted_chance, "memctrl_holds": memctrl_holds,
                                "no_weight_transport": wt, "chance": ch}
        summary["SIGNAL"] = signal
        summary["verdict"] = read
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[transitive-inference-deep-credit] {summary['verdict']}", flush=True)
    print(f"[transitive-inference-deep-credit] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
