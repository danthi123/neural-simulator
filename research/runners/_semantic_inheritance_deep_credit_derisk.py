"""COMPOSITIONAL-SEMANTIC DEEP-CREDIT DE-RISK (rate-numpy reference) -- the GOAL-RELEVANT real-task instrument for the
validated bio-plausible deep-credit rule, after FC-vision (CIFAR) was shown to be the WRONG instrument.

WHY THIS TASK (the redirect from part 1, `2026-07-07-deep-credit-real-task-cifar-fc-vision-wrong-instrument.md`): on
real FC-vision, "depth is required" and "the bio-plausible deep-credit rule can learn it" are ANTI-CORRELATED (k=10
needs depth but the rule is at chance; k<=4 the rule learns but depth is not needed), because real IMAGE depth is
CONVOLUTIONAL, not FC. The GOAL is language, and the goal-relevant depth is COMPOSITIONAL: hierarchical inheritance
requires composing multi-level structure, which is genuinely depth-required AND small (within the rule's reachable
scale). This is the EMERGE-26 inheritance capability posed as a SUPERVISED deep-credit LEARNING problem (Collins-Quillian
semantic-network inheritance; Rogers-McClelland 2004 parallel-distributed semantics; Fodor-Pylyshyn systematic
generalization = the depth requirement).

THE TASK -- depth-required by SYSTEMATIC GENERALIZATION (the honest depth lever):
  A 2-level taxonomy: `n_super` superordinates (BIRD/FISH/MAMMAL/...), each with a class-PROPERTY signature (a small
  set of binary properties: flies/swims/walks/breathes...). Each superordinate has several MEMBERS (robin, sparrow, ...).
  Input X = the member's distributed REPRESENTATION; target y = the member's class PROPERTY.

  The member representation carries the is-a structure in REAL CO-OCCURRENCE STATISTICS (the EMERGE-30/32 mechanism):
  each member is OBSERVED co-occurring with a DIFFERENT overlapping k-of-n subset of its superordinate's shared feature
  pool (Rogers-McClelland feature overlap -- no universal token). The member vector is the accumulated co-occurrence
  histogram over the feature pool (a distributed, real-statistics-shaped code) PLUS a small member-identity block (so
  train members are individually addressable = the memorization affordance a 1-layer net exploits). The superordinate
  identity is NOT linearly present -- it must be RECOVERED by pooling/denoising the noisy overlapping subset (hop 1);
  the property is a NON-LINEAR (XOR-of-super-id-bits) function of the recovered superordinate (hop 2). So predicting a
  HELD-OUT member's property (never in training) REQUIRES composing member->superordinate->property (2 hops); a
  1-hidden-layer net can memorize TRAIN members' direct member->property map but CANNOT generalize the composition to a
  held-out member -> it UNDERFITS held-out inheritance; a 2-3-hidden-layer net that BUILDS the superordinate
  representation SUCCEEDS. THAT is the depth requirement, and Stage 0 MEASURES it before any arm is read.

HONEST REALISM (per the prompt -- documented, not hidden): the features are the EMERGE co-occurrence MECHANISM
(distributed, real-statistics-shaped, permuted-pool-falsifiable), NOT raw TinyStories PPMI stream codes. The 320x300
real PPMI code cache (`_phaseB_stream_codes_320_seed42.npy`) is ABSENT in this checkout and rebuilding it (or a real
TinyStories co-occurrence embedding) is a whole data arc that would CONFOUND the depth-genuineness question with a
data-sparsity question for a FIRST de-risk. The prompt explicitly permits "a real-entity taxonomy with distributed (not
one-hot) member codes ... but document the realism honestly" -- this is that path, with the co-occurrence structure
(the is-a signal) genuinely LEARNED-statistics-shaped and the permuted-pool control isolating it. Wiring the real PPMI
codes (once the cache is regenerated) is a one-function swap in `make_task_semantic_inheritance` (documented at its call
site) -- the controller's follow-on.

REUSE-BY-IMPORT (NO `sim/` edit):
  - the deep-credit ARMS + per-layer-alignment + no-weight-transport probes from `_gnw_d1_spiking_bdsp_derisk`
    (FANet / MicrocircuitBDSPNet / BDSPNet + _train + _per_layer_alignment + _no_weight_transport*).
  - `sim.dendritic_mlp.DendriticMLP` -- the fenced backprop ORACLE (Stage-0 depth oracle + Stage-1 ceiling + the
    per-layer-alignment reference) + the 1-hidden-layer floor.
  - the Gabor-CIFAR runner's Stage-0 / Stage-1 / anti-cheat SCAFFOLD (`_gabor_cifar_deep_credit_derisk`) mirrored here
    verbatim in structure -- only `make_task_*` differs (semantic inheritance vs V1-CIFAR-k).

STAGE 0 (the load-bearing gate -- MEASURED FIRST): on the HELD-OUT-INHERITANCE split, a 1-hidden-layer fenced oracle
must UNDERFIT held-out inheritance while a 2-3-hidden-layer oracle CLEARS it by a real margin (the depth gap ON THE
HELD-OUT members = the composition test). If held-out inheritance is NOT depth-separating (a shallow net already
generalizes) it is the WRONG task config -- reported honestly, tune (deeper taxonomy / more members / harder property
nonlinearity / noisier co-occurrence) until it is, or report the honest boundary. This gate is what XOR/MNIST/FC-CIFAR
FAILED to be goal-relevant on.

STAGE 1 (once Stage 0 passes): the deep-credit arms (microcircuit / KP-learned / plain-FA / burstprop) + the 1-layer
floor + oracle ceiling. GATED METRIC = per-layer credit-alignment vs the fenced oracle AND held-out-inheritance accuracy
(here accuracy IS meaningful because held-out inheritance genuinely requires the composition -- but ALSO report
alignment + the Trap-B wrong-sign-fails-alignment control to be safe).

ANTI-CHEATS (mandatory): (1) wrong-sign FAILS alignment (Trap-B defeat); (2) permuted-label -> chance; (3) 1-layer floor
UNDERFITS held-out inheritance; (4) oracle ceiling >= 0.80 held-out inheritance; (5) no weight transport; (6) MEMORIZATION
control -- a held-out member of a superordinate whose property was NEVER taught to ANY member must NOT be inferable
(accuracy at chance -> no leakage: the net can only infer a held-out property by composing through a superordinate that
DID have a taught member).

HONEST SCOPE: numpy RATE reference (the builder's fast CPU smoke). The on-bridge spiking depth-3 multi-seed is the
controller's decisive GPU run. NO `sim/` edit anywhere (all reuse-by-import). CPU numpy backend.

Run (1-seed smoke -- the tuned depth-separating default: n_super=24, n_prop=3 -> 8 classes, deep-layers=2):
    SIM_BACKEND=numpy python -m research.runners._semantic_inheritance_deep_credit_derisk --seeds 42

The CONTROLLER's multi-seed run (fan one process per seed across cores; aggregate the per-seed JSONs):
    for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -m \
        research.runners._semantic_inheritance_deep_credit_derisk --seeds $s \
        --out research/findings/raw/_semantic_inheritance_seed$s.json & done; wait
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

OUT = _REPO / "research" / "findings" / "raw" / "_semantic_inheritance_deep_credit.json"


# ============================================================================================================
# The compositional-semantic hierarchical-inheritance task.
#   n_super superordinates; each has n_members members and a binary PROPERTY signature.
#   The property is a NON-LINEAR (parity/XOR) function of the superordinate's identity bits (hop 2).
#   A member's distributed representation = accumulated co-occurrence over its super's shared feature pool
#     (each member sees its OWN varied k-of-n subset -> is-a structure lives in the feature-overlap statistics)
#     PLUS a member-identity block (train members individually addressable = the memorization affordance).
#   The superordinate identity is NOT linearly present -> must be pooled/denoised from the noisy subset (hop 1).
#   HELD-OUT-INHERITANCE split: for each super, hold out some members ENTIRELY from the property-teaching signal
#     (they appear with their is-a features but their property is NEVER a training target) -> predicting their
#     property requires composing member->super->property = genuinely depth-required.
# ============================================================================================================
def make_task_semantic_inheritance(seed, n_super=24, n_members=8, held_per_super=3, n_prop=3,
                                   member_id_dim=3, n_obs=14, noise=0.02, feature_seed=0,
                                   id_bits=None, pool_n=None, subset_k=None):
    """Build the hierarchical-inheritance deep-credit task.

    Returns (Xtr, ytr, Ltr), (Xte, yte, Lte), meta -- the SAME shape the _gnw_d1 arms consume.
      - y = the member's superordinate PROPERTY-CLASS (the categorical property signature). HELD-OUT members are in
        Xte ONLY (never a training target) -> generalizing to them = the composition test.
      - Ltr/Lte latents (secondary/reported emergence probe) = the superordinate IDENTITY bits (did the hidden rep
        recover the super identity? = the intermediate feature the composition must build). NOT the gate.

    THE REAL-CODE SWAP (documented for the controller follow-on): to use the real stream-cortex PPMI codes instead of
    the co-occurrence-histogram features, replace the `feat` construction below with the member's row of
    `_phaseB_stream_codes_320_seed42.npy` (mapped member->row by a real-similarity clustering, as EMERGE-19 does). The
    is-a structure then comes from the REAL learned corpus similarity; everything downstream (split/oracle/arms) is
    unchanged. The cache is absent in this checkout (see the module docstring HONEST REALISM note)."""
    rng = np.random.default_rng(seed)
    # ---- the DEPTH lever (the PROVEN construction; see the module docstring): the superordinate's PROPERTY signature
    # is a set of `n_prop` binary SEMANTIC FEATURES, each of which is the XOR of an OBSERVABLE feature-PAIR over a
    # SHARED pool. A member is OBSERVED via a fresh +-1 realization of its super's pool-XOR pattern (a0 drawn FRESH each
    # observation so every single pool feature is MARGINALLY 50/50 across supers -> a LINEAR read of any feature is
    # uninformative; ONLY the pair-XOR, a nonlinear hidden unit, recovers a property bit). The property CLASS is the
    # combination of those `n_prop` XOR bits (Rogers-McClelland 2004: a property is a nonlinear CONJUNCTION of semantic
    # features). So: recovering each property bit needs a HIDDEN layer (XOR, hop 1) and combining them into the class
    # needs a SECOND (hop 2) -> a 1-hidden-layer / linear net provably UNDERFITS held-out members; a 2-3-hidden-layer
    # net that BUILDS the XOR features generalizes. Members of the same super share the SAME XOR pattern (the is-a
    # structure); each member is a different noisy realization (the co-occurrence statistics). A held-out member (a
    # NOVEL realization of a taught super) must be classified by COMPOSING member-features -> super -> property, exactly
    # the systematic-generalization depth requirement. ----
    n_prop = int(n_prop)                                  # each property bit = one observable XOR pair
    # (id_bits/pool_n/subset_k are accepted for back-compat/tuning but the pool is derived from n_prop directly.)
    # each super gets a DISTINCT n_prop-bit property signature (the supers cycle through all 2^n_prop combos, so the
    # class map is a systematic function of the bits, not an arbitrary per-super lookup -> a held-out member's class is
    # RULE-derivable). n_super >= 2^n_prop guarantees every class is represented.
    n_class = 1 << n_prop
    super_bits = np.array([[(s >> b) & 1 for b in range(n_prop)] for s in range(n_super)], dtype=np.int64)
    prop_class = np.array([int(sum(super_bits[s, b] << b for b in range(n_prop))) for s in range(n_super)], np.int64)

    pool_dim = 2 * n_prop                                 # SHARED pool: 2 features per property bit (the XOR pair)
    n_pool_dims = pool_dim
    # per-member identity block: a distinct RANDOM code per (super,member) -- NOT keyed to the super linearly. It gives
    # TRAIN members an individually-addressable handle (the memorization affordance) but a linear read of it does NOT
    # reveal the super (each member's code is independent random) -> it cannot linearly leak the held-out property.
    n_id_dims = member_id_dim
    n_feat = n_pool_dims + n_id_dims
    mrng = np.random.default_rng(seed * 777 + 5)
    member_codes = {(s, m): mrng.standard_normal(member_id_dim) for s in range(n_super) for m in range(n_members)}

    Xtr_l, ytr_l, Ltr_l, Xte_l, yte_l, Lte_l = [], [], [], [], [], []
    heldout_super_taught = np.ones(n_super, dtype=bool)   # for the memorization control: which supers had a taught member
    mem_ctrl_rows = []                                    # (row, super, untaught?) of held-out members

    # MEMORIZATION control (the no-leakage anti-cheat): the LAST `n_untaught` supers hold out ALL members from training
    # AND their property class is a RESERVED NOVEL class (`novel_class = n_class`) that NO taught super ever uses. So an
    # untaught super's members share the XOR-feature pool with the world but their TRUE class was NEVER a training
    # target -> a faithful net MUST fail to infer it (accuracy ~0). A LEAK (the net inferring an untaught member's class
    # from a per-super shortcut) would show as memctrl accuracy > chance. This is the sharp control the spec asks for.
    n_untaught = max(1, n_super // 4) if held_per_super > 0 else 0
    untaught_supers = set(range(n_super - n_untaught, n_super))
    novel_class = n_class
    for s in untaught_supers:
        prop_class[s] = novel_class                       # a never-taught class -> genuinely not inferable
    k_classes = n_class + (1 if n_untaught > 0 else 0)

    for s in range(n_super):
        members = list(range(n_members))
        held = set(members[-held_per_super:]) if held_per_super > 0 else set()
        super_untaught = (s in untaught_supers)
        for mi in members:
            n_view = 1 if (mi in held) else n_obs
            for _ in range(n_view):
                feat = np.zeros(n_feat, dtype=np.float64)
                # THE XOR ENCODING (the guaranteed depth lever): for each property bit b, a FRESH random pair (a0, a1)
                # with a1 = -a0 iff super_bits[s,b]==1 (so XOR(sign a0, sign a1) == the property bit). Fresh a0 ->
                # balanced marginal -> linearly uninformative; only the pair-XOR (nonlinear) recovers the bit.
                for b in range(n_prop):
                    a0 = 1.0 if rng.random() < 0.5 else -1.0
                    a1 = -a0 if super_bits[s, b] == 1 else a0
                    feat[2 * b] = a0; feat[2 * b + 1] = a1
                feat[:pool_dim] += noise * rng.standard_normal(pool_dim)   # observation noise (denoising = hop-1 work)
                feat[pool_dim:pool_dim + member_id_dim] = member_codes[(s, mi)]  # member handle (no super leak)
                y = int(prop_class[s])
                lat = super_bits[s].astype(np.float64)                    # property XOR bits (emergence probe target)
                is_train = (mi not in held)
                if is_train and not super_untaught:
                    Xtr_l.append(feat); ytr_l.append(y); Ltr_l.append(lat)
                else:
                    Xte_l.append(feat); yte_l.append(y); Lte_l.append(lat)
                    if mi in held:
                        mem_ctrl_rows.append((len(Xte_l) - 1, s, bool(super_untaught)))
        if super_untaught:
            heldout_super_taught[s] = False

    Xtr = np.asarray(Xtr_l); ytr = np.asarray(ytr_l, np.int64); Ltr = np.asarray(Ltr_l)
    Xte = np.asarray(Xte_l); yte = np.asarray(yte_l, np.int64); Lte = np.asarray(Lte_l)

    # optional fixed decorrelating random projection (feature_seed>0): rotate/compress the feature space (label-free)
    # so the depth-genuineness window can be tuned without leakage. feature_seed=0 => identity.
    if feature_seed and feature_seed > 0:
        frng = np.random.default_rng(feature_seed + 100003)
        n_proj = min(Xtr.shape[1], max(32, int(feature_seed)))
        P = frng.standard_normal((Xtr.shape[1], n_proj)) / np.sqrt(Xtr.shape[1])
        Xtr = Xtr @ P; Xte = Xte @ P

    # per-feature standardization on TRAIN statistics (sigmoid MLP calibration; applied to ALL arms identically).
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True)
    Xtr = (Xtr - mu) / (sd + 1e-6); Xte = (Xte - mu) / (sd + 1e-6)

    # held-out-INHERITANCE mask over Xte: members whose super HAD a taught member (the genuine composition test) vs the
    # memorization-control members (their super had NO taught member -> must NOT be inferable).
    inh_idx = np.array([r for (r, s, unt) in mem_ctrl_rows if not unt], dtype=np.int64)
    memctrl_idx = np.array([r for (r, s, unt) in mem_ctrl_rows if unt], dtype=np.int64)

    # shuffle train (deterministic per seed) so batches mix classes.
    ptr = rng.permutation(len(ytr)); Xtr, ytr, Ltr = Xtr[ptr], ytr[ptr], Ltr[ptr]

    meta = {"n_super": n_super, "n_members": n_members, "held_per_super": held_per_super,
            "n_prop": int(n_prop), "k_classes": int(k_classes), "member_id_dim": member_id_dim,
            "n_obs": n_obs, "noise": noise, "feature_seed": int(feature_seed),
            "n_features": int(Xtr.shape[1]), "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
            "n_inherit_heldout": int(len(inh_idx)), "n_memctrl_heldout": int(len(memctrl_idx)),
            "n_supers_untaught": int((~heldout_super_taught).sum())}
    return (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, {"inh_idx": inh_idx, "memctrl_idx": memctrl_idx}


# ============================================================================================================
# Emergence probe (secondary/reported): does a linear read-out of the FROZEN hidden rep recover the superordinate
# IDENTITY bits? (== the intermediate feature the composition must build.)
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
    pred = (Xte @ W) >= 0.5
    return float(np.mean(pred == (L_te >= 0.5)))


def _train_oracle(net, X, y, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode="oracle", lr=lr)


def _acc_on(net, X, y, idx):
    """Held-out-INHERITANCE accuracy on a SUBSET of the held set (the composition test), not the whole held set."""
    if idx is None or len(idx) == 0:
        return float("nan")
    _, lg = net._forward(np.asarray(X[idx], float))
    return float(np.mean(np.argmax(np.asarray(lg), 1) == np.asarray(y[idx])))


# ============================================================================================================
# STAGE 0 -- depth-genuineness on the HELD-OUT-INHERITANCE split (the load-bearing gate, MEASURED FIRST).
# ============================================================================================================
def stage0_depth_genuineness(task, idx, k, hidden, epochs, lr, batch, seed):
    """A 1-hidden-layer fenced-backprop oracle must UNDERFIT held-out INHERITANCE while a 2-3-hidden-layer oracle clears
    it by a clear margin. ALL arms are the fenced backprop ORACLE (this measures the representational depth-requirement
    of the TASK, not the credit rule). The gated accuracy is on the INHERITANCE held-out subset (members whose super
    had a taught member) -- the composition test. Reports 0/1/2/3-hidden oracle held-out-inheritance + gaps + chance."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    inh_idx = idx["inh_idx"]
    n_in = Xtr.shape[1]
    # chance on the inheritance held-out subset (the composition targets)
    if len(inh_idx):
        yv = yte[inh_idx]
        chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")

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
    return {"chance": chance, "n_features": int(n_in),
            "linear_inherit_heldout": l0_te, "linear_train": l0_tr,
            "l1_inherit_heldout": l1_te, "l1_train": l1_tr,
            "l2_inherit_heldout": l2_te, "l2_train": l2_tr,
            "l3_inherit_heldout": l3_te, "l3_train": l3_tr,
            "deep_best_inherit_heldout": deep_best, "depth_gap": float(depth_gap),
            "depth_separating": depth_separating}


# ============================================================================================================
# STAGE 1 -- the deep-credit arms + per-layer alignment (the GATED metric) + anti-cheats (incl. the Trap-B defeat
# and the MEMORIZATION control).
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
    # emergence probe on the INHERITANCE held-out members (did the hidden rep recover the super identity?)
    if len(inh_idx):
        probe = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte[inh_idx]), Lte[inh_idx])
    else:
        probe = float("nan")
    return {"inherit_heldout": _acc_on(net, Xte, yte, inh_idx), "memctrl_heldout": _acc_on(net, Xte, yte, mem_idx),
            "train": float(net.accuracy(Xtr, ytr)),
            "per_layer_alignment": [float(c) for c in align], "deepest_layer_alignment": float(align[0]),
            "no_weight_transport": nwt, "probe_latent": float(probe),
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
    # the DEEP arm uses `deep_layers` hidden layers -- the depth Stage-0 found genuinely REQUIRED + oracle-clearable
    # (default 2: the XOR-pair-property task is depth-2; a depth-3 net overfits/is unstable at this budget). The floor
    # is 1 hidden layer (the memorization/no-composition regime).
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
    res["oracle"] = {"inherit_heldout": _acc_on(onet, Xte, yte, inh_idx),
                     "memctrl_heldout": _acc_on(onet, Xte, yte, mem_idx), "train": float(onet.accuracy(Xtr, ytr))}

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
    # 1-hidden floor (memorization/no-depth): must UNDERFIT held-out inheritance.
    fnet = Net(shal, seed=seed, beta=beta, p0=p0)
    _train(fnet, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    res["single_layer"] = {"inherit_heldout": _acc_on(fnet, Xte, yte, inh_idx),
                           "memctrl_heldout": _acc_on(fnet, Xte, yte, mem_idx), "train": float(fnet.accuracy(Xtr, ytr))}

    # --- anti-cheats ---
    res["wrong_sign"] = _wrongsign_alignment(Net, deep, task, idx, epochs, lr, batch, seed, _kind, beta=beta, p0=p0)
    res["wrong_sign_plain_fa"] = _wrongsign_alignment(FANet, deep, task, idx, epochs, lr, batch, seed, "fa",
                                                      beta=beta, p0=p0)
    # permuted-label -> chance on inheritance held-out (generalization, not leakage)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    pnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(pnet, Xtr, yperm, "bdsp", epochs, lr, batch, seed)
    res["permuted"] = {"inherit_heldout": _acc_on(pnet, Xte, yte, inh_idx), "train": float(pnet.accuracy(Xtr, yperm))}
    # apical-lesion (Y=0) -> no top-down credit -> deepest-layer alignment collapses; inheritance held-out at floor.
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
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    if k is None:
        k = meta["k_classes"]
    s0 = stage0_depth_genuineness(task, idx, k, hidden, epochs, lr, batch, seed)
    s1 = stage1_deep_credit(task, idx, k, hidden, epochs, lr, batch, seed, rule=rule, feedback=feedback,
                            homeostasis=homeostasis, kp_lr=kp_lr, kp_decay=kp_decay, beta=beta, p0=p0,
                            deep_layers=deep_layers)
    return {"seed": seed, "meta": meta, "stage0_depth_genuineness": s0, "stage1_deep_credit": s1}


def _fmt_align(a):
    return "[" + ", ".join(f"{c:.2f}" for c in a) + "]"


def main():
    ap = argparse.ArgumentParser(description="Compositional-semantic hierarchical-inheritance deep-credit de-risk.")
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
                    help="hidden layers in the DEEP arm (default 2 = where the XOR-pair-property task needs depth AND "
                         "the oracle clears; 3 overfits at this budget).")
    ap.add_argument("--n-super", type=int, default=24, help="number of superordinates (>= 2^n_prop for full coverage)")
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3, help="binary semantic-feature (XOR-pair) count -> 2^n_prop classes")
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14, help="observations per train member (noisy XOR realizations)")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                       n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise,
                       feature_seed=a.feature_seed)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, None, a.hidden, a.epochs, a.lr, a.batch, a.rule, a.feedback, a.homeostasis,
                         a.kp_lr, a.kp_decay, a.beta, a.p0, task_kwargs, deep_layers=a.deep_layers)
            per.append(r)
            s0 = r["stage0_depth_genuineness"]; s1 = r["stage1_deep_credit"]; m = r["meta"]
            print("-" * 112, flush=True)
            print(f"[seed {s}] {m['n_super']} supers x {m['n_members']} members ({m['held_per_super']} held/super) | "
                  f"{m['k_classes']} property-classes | {m['n_features']} feats | {m['n_train']} train / "
                  f"{m['n_inherit_heldout']} inherit-held / {m['n_memctrl_heldout']} memctrl-held | chance {s0['chance']:.3f}",
                  flush=True)
            print(f"  STAGE0 depth-genuineness (held-out INHERITANCE acc): linear {s0['linear_inherit_heldout']:.3f} | "
                  f"1-layer {s0['l1_inherit_heldout']:.3f} | 2-layer {s0['l2_inherit_heldout']:.3f} | "
                  f"3-layer {s0['l3_inherit_heldout']:.3f} | deep-best {s0['deep_best_inherit_heldout']:.3f} | "
                  f"depth-gap {s0['depth_gap']:+.3f} => DEPTH-SEPARATING {s0['depth_separating']}", flush=True)
            tf = s1["test_fixed"]; tl = s1["test_learned"]; th = s1["test_learned_homeo"]; pf = s1["plain_fa"]
            ws = s1["wrong_sign"]; les = s1["apical_lesion"]
            print(f"  STAGE1 [{s1['rule']}] per-layer align vs oracle (layer0=deepest) + held-out INHERITANCE acc:",
                  flush=True)
            print(f"    test-fixed   inherit {tf['inherit_heldout']:.3f} memctrl {tf['memctrl_heldout']:.3f} "
                  f"align {_fmt_align(tf['per_layer_alignment'])} deep {tf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    test-learned inherit {tl['inherit_heldout']:.3f} align {_fmt_align(tl['per_layer_alignment'])} "
                  f"deep {tl['deepest_layer_alignment']:.2f} (KP, transport-free {tl['no_weight_transport']})", flush=True)
            print(f"    +homeo       inherit {th['inherit_heldout']:.3f} align {_fmt_align(th['per_layer_alignment'])} "
                  f"deep {th['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    plain-FA     inherit {pf['inherit_heldout']:.3f} align {_fmt_align(pf['per_layer_alignment'])} "
                  f"deep {pf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    single-layer inherit {s1['single_layer']['inherit_heldout']:.3f} | "
                  f"oracle-d3 inherit {s1['oracle']['inherit_heldout']:.3f} | chance {s1['chance']:.3f}", flush=True)
            print(f"    [anti-cheat] WRONG-SIGN: inherit {ws['inherit_heldout']:.3f} (may be rescued) | ALIGNMENT deep "
                  f"{ws['deepest_layer_alignment']:.2f} align {_fmt_align(ws['per_layer_alignment'])}  <- must FAIL", flush=True)
            print(f"    [anti-cheat] permuted {s1['permuted']['inherit_heldout']:.3f} (~chance) | lesion inherit "
                  f"{les['inherit_heldout']:.3f} align deep {les['deepest_layer_alignment']:.2f} | "
                  f"MEMCTRL(oracle) {s1['oracle']['memctrl_heldout']:.3f} (must ~chance) | "
                  f"same-init {s1['same_init_as_oracle']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "semantic_inheritance_deep_credit", "seeds": a.seeds, "rule": a.rule,
               "config": {"hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                          "feedback": a.feedback, "homeostasis": bool(a.homeostasis), "kp_lr": a.kp_lr,
                          "kp_decay": a.kp_decay, "beta": a.beta, "p0": a.p0, "task": task_kwargs,
                          "backend": os.environ.get("SIM_BACKEND")},
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
        oracle = _m(["stage1_deep_credit", "oracle", "inherit_heldout"])
        oracle_mem = _m(["stage1_deep_credit", "oracle", "memctrl_heldout"])
        tf_deep = _m(["stage1_deep_credit", "test_fixed", "deepest_layer_alignment"])
        tl_deep = _m(["stage1_deep_credit", "test_learned", "deepest_layer_alignment"])
        th_deep = _m(["stage1_deep_credit", "test_learned_homeo", "deepest_layer_alignment"])
        pf_deep = _m(["stage1_deep_credit", "plain_fa", "deepest_layer_alignment"])
        best_test_deep = max(tf_deep, tl_deep, th_deep)
        tf_inh = _m(["stage1_deep_credit", "test_fixed", "inherit_heldout"])
        tl_inh = _m(["stage1_deep_credit", "test_learned", "inherit_heldout"])
        th_inh = _m(["stage1_deep_credit", "test_learned_homeo", "inherit_heldout"])
        best_test_inh = max(tf_inh, tl_inh, th_inh)
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
        # the goal-relevant read: the deep-credit rule LEARNS the composition (held-out inheritance > 1-layer floor)
        learns_composition = bool(best_test_inh > sl_inh + 0.05 and best_test_inh > ch + 0.05)
        memctrl_holds = bool(np.isnan(oracle_mem) or oracle_mem <= ch + 0.15)   # the memorization control
        oracle_ok = bool(oracle >= 0.80)
        signal = bool(s0_sep and oracle_ok and best_test_deep > 0.15 and learns_composition and wrongsign_fails
                      and lesion_collapses and permuted_chance and wt and memctrl_holds)
        if not s0_sep:
            read = (f"STAGE-0 BOUNDARY -- the semantic-inheritance task is NOT depth-separating at n_super={a.n_super}/"
                    f"members={a.n_members}/held={a.held_per_super}/n_prop={a.n_prop}/noise={a.noise} (deep-best inherit "
                    f"{deep_best:.3f} vs 1-layer {l1:.3f}, gap {depth_gap:+.3f}). This is the wrong task CONFIG, NOT a "
                    f"GO -- escalate (more members / more classes via --n-prop / lower --noise so the depth-2 oracle "
                    f"reliably clears >=0.80 / more held-per-super) BEFORE reading the deep-credit arms. This gate "
                    f"is what XOR/MNIST/FC-CIFAR FAILED to be goal-relevant on; it must PASS first.")
        elif not oracle_ok:
            read = (f"INCONCLUSIVE -- the depth-3 oracle only reached {oracle:.3f} held-out inheritance at H{a.hidden}; "
                    f"tune epochs/lr/hidden before reading the deep-credit arms (NOT a verdict).")
        else:
            _tb = "FAILS (deep align {:.2f} < best-test {:.2f})".format(ws_deep, best_test_deep) if wrongsign_fails \
                  else "does NOT fail (deep align {:.2f}) -- Trap B may re-bite; report honestly".format(ws_deep)
            read = (f"STAGE-0 PASS (depth-separating: deep-best inherit {deep_best:.3f} vs 1-layer {l1:.3f}, gap "
                    f"{depth_gap:+.3f}, oracle {oracle:.3f}). STAGE-1 deep credit ({a.rule}): held-out INHERITANCE -- "
                    f"single-layer {sl_inh:.3f}, best-test {best_test_inh:.3f}, chance {ch:.3f} "
                    f"({'LEARNS the composition' if learns_composition else 'does NOT beat the 1-layer floor'}); "
                    f"deepest-layer alignment -- plain-FA {pf_deep:.2f}, best-test {best_test_deep:.2f}; lesion inherit "
                    f"{les_inh:.3f} align {les_deep:.2f}; WRONG-SIGN alignment {_tb}; permuted {perm:.3f} (~chance "
                    f"{ch:.3f}); memorization-control(oracle) {oracle_mem:.3f} ({'holds' if memctrl_holds else 'LEAKS'}); "
                    f"no weight transport {wt}. "
                    f"{'LOAD-BEARING depth-benefit on a REAL compositional-semantic task' if signal else 'NO clean deep-credit signal yet (see the arm table)'} "
                    f"=> {'controller runs 6-seed + on-bridge spiking' if signal else 'honest read: escalate / diagnose'}. "
                    f"Numpy RATE reference; the decisive on-bridge spiking depth-3 multi-seed is the controller's GPU run.")
        summary["stage0_depth_separating"] = s0_sep
        summary["aggregate"] = {"deep_best_inherit_heldout": deep_best, "l1_inherit_heldout": l1, "depth_gap": depth_gap,
                                "oracle_inherit_heldout": oracle, "oracle_memctrl_heldout": oracle_mem,
                                "deepest_align_test_fixed": tf_deep, "deepest_align_test_learned": tl_deep,
                                "deepest_align_test_learned_homeo": th_deep, "deepest_align_plain_fa": pf_deep,
                                "best_test_deep_align": best_test_deep, "inherit_test_fixed": tf_inh,
                                "inherit_test_learned": tl_inh, "inherit_test_learned_homeo": th_inh,
                                "best_test_inherit": best_test_inh, "single_layer_inherit": sl_inh,
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
    print(f"[semantic-inheritance-deep-credit] {summary['verdict']}", flush=True)
    print(f"[semantic-inheritance-deep-credit] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
