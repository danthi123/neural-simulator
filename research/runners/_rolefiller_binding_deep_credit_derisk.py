"""ROLE-FILLER BINDING / SYSTEMATIC-RECOMBINATION DEEP-CREDIT DE-RISK (rate-numpy reference) -- the CULMINATING,
most-language-relevant real-task instrument for the validated bio-plausible deep-credit rule.

WHY THIS TASK (the arc's conclusion, `2026-07-08-deep-credit-depth-lives-in-nonlinear-conjunction-not-natural-shortcuts.md`):
three self-correcting Stage-0 negatives established that the "natural" tasks are shortcut-able (perception=convolutional,
category=linear-embedding, order=scalar-score) and that the ONLY depth-required task was a NONLINEAR CONJUNCTION (part-2
XOR-over-pool). The pattern predicts supervised deep-credit's depth-benefit needs a target that is a nonlinear
conjunction / binding -- and ROLE-FILLER BINDING (Fodor-Pylyshyn systematic recombination) is the language-relevant such
task: the answer depends on the role x filler CONJUNCTION (not a per-item score, not linear), and systematic
generalization to NOVEL (role, filler) combos requires learning the bind, not memorizing seen bindings. This is the exact
VSA-composer / EMERGE binding capability (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`)
posed as a SUPERVISED deep-credit LEARNING problem.

THE TASK -- systematic recombination / role-filler binding:
  R roles x F fillers, each an ARBITRARY random +-1 code. A "fact" superposes a role code and a filler code (concatenated
  blocks) plus a small per-item identity handle (train combos individually addressable = the memorization affordance).
  The TARGET is the VALUE CLASS of the (role, filler) CONJUNCTION -- a nonlinear function of the SPECIFIC role x filler
  pairing (per-property XOR / AND / conditional of a role key-bit and a filler key-bit, each key-bit itself recovered as a
  parity over a noisy code subset -> hop 1 recovery, hop 2 bind). Neither role nor filler alone determines the value; a
  LINEAR decode is at chance (verified by the Stage-0 linear probe). TRAIN on a SUBSET of (role, filler) combos; TEST
  HELD-OUT NOVEL (role, filler) combos never seen together. A net that MEMORIZES seen bindings cannot systematically
  generalize to novel combos; one that LEARNS the role(x)filler bind (a conjunctive interaction) generalizes.

STAGE 0 (the SELF-CORRECTING gate -- MEASURED FIRST, the load-bearing question): is held-out NOVEL-combination binding
  DEPTH-REQUIRED? A 1-hidden-layer fenced-backprop oracle must UNDERFIT held-out novel combos while a 2-3-hidden oracle
  CLEARS it by a real margin AND the LINEAR probe on the codes is near chance for held-out novel combos (not
  linearly/score shortcut-able). If NOT depth-separating, report the honest boundary -- the arc pattern would then be:
  EVEN binding-as-supervised-classification is single-hidden-layer-solvable (universal approximation lets one wide hidden
  layer form the role x filler conjunctive feature directly), and the depth-benefit lives in the SEQUENTIAL / recursive
  composition (a multi-hop bind), not a single role-filler bind. A built-in POSITIVE CONTROL (the part-2 XOR-over-pool
  target, the one arc task that IS depth-required) proves the harness CAN detect depth when present -> makes any negative
  trustworthy.

STAGE 1 (ONLY if Stage 0 passes): the deep-credit arms (microcircuit / KP-learned / plain-FA / burstprop) + the 1-layer
  floor + the depth-oracle ceiling. GATED METRIC = held-out NOVEL-combination accuracy (> the 1-layer floor) AND per-layer
  credit-alignment vs the fenced oracle. Anti-cheats: (1) wrong-sign FAILS alignment (Trap-B); (2) permuted-label ->
  chance; (3) 1-layer floor UNDERFITS held-out novel combos; (4) oracle ceiling >= 0.80; (5) no weight transport;
  (6) MEMORIZATION control -- a held-out combo whose role OR filler NEVER appeared in ANY training binding is un-inferable
  (accuracy at chance -> the net can only infer a held-out combo by BINDING through role/filler each seen in some training
  combo, not by a per-combo lookup).

REUSE-BY-IMPORT (NO `sim/` edit): the deep-credit ARMS + per-layer alignment + no-weight-transport probes from
  `_gnw_d1_spiking_bdsp_derisk`; `sim.dendritic_mlp.DendriticMLP` (fenced backprop ORACLE + 1-hidden floor); the
  Stage-0/Stage-1/anti-cheat SCAFFOLD mirrored from `_semantic_inheritance_deep_credit_derisk` VERBATIM in structure --
  only `make_task_*` differs (role-filler binding vs semantic inheritance).

HONEST SCOPE: numpy RATE reference (the builder's fast CPU smoke). The on-bridge spiking depth-3 multi-seed is the
  controller's decisive GPU run. NO `sim/` edit anywhere (all reuse-by-import). CPU numpy backend.

Run (1-seed smoke):
    SIM_BACKEND=numpy python -m research.runners._rolefiller_binding_deep_credit_derisk --seeds 42

The CONTROLLER's multi-seed run (fan one process per seed across cores; aggregate the per-seed JSONs):
    for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -m \
        research.runners._rolefiller_binding_deep_credit_derisk --seeds $s \
        --out research/findings/raw/_rolefiller_binding_seed$s.json & done; wait
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

OUT = _REPO / "research" / "findings" / "raw" / "_rolefiller_binding_deep_credit.json"


# ============================================================================================================
# The role-filler binding / systematic-recombination task.
#   R roles x F fillers; each an ARBITRARY random +-1 code carrying `key_bits` recoverable KEY bits (each = the
#     SIGN-PARITY over `sub_k` code dims, realized FRESH per observation so a single dim is marginally 50/50 ->
#     LINEARLY uninformative; only the subset-parity, a nonlinear hidden unit, recovers a key bit = hop 1).
#   A "fact" X = [role_code, fill_code, role_id, fill_id] (the id blocks = the memorization handle for TRAIN combos).
#   TARGET y = the VALUE CLASS of the (role, filler) CONJUNCTION: per property b, a `bind_op` gate of the recovered
#     role key-bit and filler key-bit (XOR / AND / MUX) -> the value depends on BOTH (not role or filler alone) and
#     is NONLINEAR (linear probe at chance) = hop 2 bind.
#   HELD-OUT-NOVEL-COMBINATION split: hold out a random SUBSET of (role, filler) combos ENTIRELY from training. A
#     held-out combo whose role AND filler each appear in >=1 TRAIN combo = the genuine BINDING test (infer by
#     recombination). A held-out combo whose role OR filler NEVER appears in ANY train combo = the MEMORIZATION
#     control (un-inferable -> must be at chance).
# ============================================================================================================
def make_task_rolefiller_binding(seed, n_roles=14, n_fillers=14, key_bits=2, sub_k=2, n_obs=18,
                                 held_frac=0.30, noise=0.03, id_dim=3, bind_op="xor",
                                 memctrl_holdouts=2, feature_seed=0):
    """Build the role-filler binding deep-credit task.

    Returns (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx -- the SAME shape the _gnw_d1 arms consume (idx =
      {"inh_idx": novel-COMBINATION held-out rows (the binding test), "memctrl_idx": un-inferable held-out rows}).
      - y = the (role, filler) VALUE CLASS (the conjunction). HELD-OUT combos are in Xte ONLY (never a train target)
        -> generalizing to a NOVEL combo = the systematic-recombination test.
      - Ltr/Lte latents (secondary/reported emergence probe) = the recovered role+filler KEY bits concatenated
        (did the hidden rep recover the bindable bits? = the intermediate feature the bind must build). NOT the gate.

    THE DEPTH LEVER (documented): recovering each key bit = a parity over a noisy code subset (hop 1, needs a hidden
      unit); the value = a `bind_op` gate of the recovered role-bit and filler-bit (hop 2 conjunction). A held-out
      NOVEL combo (a role and filler each seen in SOME train combo, but never TOGETHER) must be classified by
      COMPOSING role-features (x) filler-features -> value, exactly the Fodor-Pylyshyn systematic-generalization depth
      requirement. Whether it is GENUINELY depth-2 (a 1-hidden net underfits) is what Stage 0 MEASURES -- a single
      role x filler bind may be single-hidden-layer-representable (universal approximation over disjoint code blocks),
      in which case Stage 0 correctly reports the honest boundary.

    THE REAL-CODE SWAP (documented for the controller follow-on): to use the real stream-cortex PPMI codes for the
      role/filler codes instead of the parity-code features, replace the `realize` blocks below with rows of
      `_phaseB_stream_codes_320_seed42.npy` mapped role/filler->row (as EMERGE-19 does). The bind structure then rides
      the REAL learned corpus codes; the split/oracle/arms are unchanged. The cache is absent in this checkout."""
    rng = np.random.default_rng(seed)
    key_bits = int(key_bits); sub_k = int(sub_k)
    n_class = 1 << key_bits
    role_key = rng.integers(0, 2, size=(n_roles, key_bits))
    fill_key = rng.integers(0, 2, size=(n_fillers, key_bits))
    # per-item identity handle: a distinct RANDOM code per role / filler (NOT keyed to the value linearly). Gives TRAIN
    # combos an individually-addressable handle (the memorization affordance) but a linear read cannot reveal the value.
    irng = np.random.default_rng(seed * 777 + 5)
    role_id = irng.standard_normal((n_roles, id_dim))
    fill_id = irng.standard_normal((n_fillers, id_dim))

    def realize(key, r):
        """`key_bits` groups of `sub_k` +-1 dims whose SIGN-PARITY encodes each key bit; FRESH per observation
        (marginal 50/50 -> linearly uninformative; recovering the bit needs the nonlinear subset-parity)."""
        out = np.zeros(len(key) * sub_k)
        for b in range(len(key)):
            vals = r.choice([-1.0, 1.0], size=sub_k)
            want = -1.0 if key[b] == 1 else 1.0                # parity==1 -> product -1
            if np.prod(np.sign(vals)) != want:
                vals[0] = -vals[0]
            out[b * sub_k:(b + 1) * sub_k] = vals
        return out

    def bind_value(rr, ff):
        """The (role, filler) VALUE CLASS = the per-property CONJUNCTION of the recovered role/filler key bits."""
        bits = []
        for b in range(key_bits):
            a = int(role_key[rr, b]); c = int(fill_key[ff, b])
            if bind_op == "and":
                bits.append(a & c)
            elif bind_op == "mux":                              # role bit gates which filler bit is read
                other = int(fill_key[ff, (b + 1) % key_bits])
                bits.append(other if a == 1 else c)
            else:                                               # "xor" (default)
                bits.append(a ^ c)
        return int(sum(bits[b] << b for b in range(key_bits)))

    # HELD-OUT-NOVEL-COMBINATION split. First choose `memctrl_holdouts` roles that are held out of EVERY training
    # combo (their fillers still train) -> a held-out combo using one of THESE roles is the MEMORIZATION control
    # (its role never appears in any train binding -> un-inferable by recombination). Then hold out a random subset
    # of the REMAINING (role, filler) combos as the NOVEL-COMBINATION binding test (both role+filler DO appear in
    # some train combo, just never together).
    memctrl_holdouts = int(min(max(0, memctrl_holdouts), max(0, n_roles - 1)))
    memctrl_roles = set(range(n_roles - memctrl_holdouts, n_roles)) if memctrl_holdouts > 0 else set()

    all_combos = [(r, f) for r in range(n_roles) for f in range(n_fillers)]
    # eligible-for-novel = combos whose role is NOT a memctrl role (those are always held out below)
    eligible = [(r, f) for (r, f) in all_combos if r not in memctrl_roles]
    perm = rng.permutation(len(eligible))
    n_held = int(len(eligible) * held_frac)
    novel_held = set(eligible[i] for i in perm[:n_held])

    # a combo is TRAIN iff its role is a normal role AND it is not in the novel-held set.
    def is_train(r, f):
        return (r not in memctrl_roles) and ((r, f) not in novel_held)

    train_roles = set(r for (r, f) in all_combos if is_train(r, f))
    train_fills = set(f for (r, f) in all_combos if is_train(r, f))

    Xtr_l, ytr_l, Ltr_l, Xte_l, yte_l, Lte_l = [], [], [], [], [], []
    mem_rows = []   # (row_in_Xte, both_present_bool) for held-out combos

    for (r, f) in all_combos:
        train = is_train(r, f)
        held = not train
        both_present = (r in train_roles) and (f in train_fills)   # inferable-by-binding iff both appear in some train combo
        n_view = 1 if held else n_obs
        for _ in range(n_view):
            rc = realize(role_key[r], rng) + noise * rng.standard_normal(key_bits * sub_k)
            fc = realize(fill_key[f], rng) + noise * rng.standard_normal(key_bits * sub_k)
            feat = np.concatenate([rc, fc, role_id[r], fill_id[f]])
            y = bind_value(r, f)
            lat = np.concatenate([role_key[r], fill_key[f]]).astype(np.float64)   # recovered-bit emergence probe target
            if train:
                Xtr_l.append(feat); ytr_l.append(y); Ltr_l.append(lat)
            else:
                Xte_l.append(feat); yte_l.append(y); Lte_l.append(lat)
                mem_rows.append((len(Xte_l) - 1, bool(both_present)))

    Xtr = np.asarray(Xtr_l); ytr = np.asarray(ytr_l, np.int64); Ltr = np.asarray(Ltr_l)
    Xte = np.asarray(Xte_l); yte = np.asarray(yte_l, np.int64); Lte = np.asarray(Lte_l)

    # optional fixed decorrelating random projection (feature_seed>0): rotate/compress (label-free); 0 => identity.
    if feature_seed and feature_seed > 0:
        frng = np.random.default_rng(feature_seed + 100003)
        n_proj = min(Xtr.shape[1], max(32, int(feature_seed)))
        P = frng.standard_normal((Xtr.shape[1], n_proj)) / np.sqrt(Xtr.shape[1])
        Xtr = Xtr @ P; Xte = Xte @ P

    # per-feature standardization on TRAIN statistics (sigmoid MLP calibration; applied to ALL arms identically).
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True)
    Xtr = (Xtr - mu) / (sd + 1e-6); Xte = (Xte - mu) / (sd + 1e-6)

    # inh_idx = NOVEL-combination held-out rows whose role+filler BOTH appear in some train combo (the binding test).
    # memctrl_idx = held-out rows whose role NEVER appears in any train combo (the un-inferable memorization control).
    inh_idx = np.array([r for (r, both) in mem_rows if both], dtype=np.int64)
    memctrl_idx = np.array([r for (r, both) in mem_rows if not both], dtype=np.int64)

    # shuffle train (deterministic per seed) so batches mix classes.
    ptr = rng.permutation(len(ytr)); Xtr, ytr, Ltr = Xtr[ptr], ytr[ptr], Ltr[ptr]

    meta = {"n_roles": n_roles, "n_fillers": n_fillers, "key_bits": key_bits, "sub_k": sub_k,
            "k_classes": int(n_class), "bind_op": bind_op, "id_dim": id_dim, "n_obs": n_obs, "noise": noise,
            "held_frac": held_frac, "memctrl_holdouts": memctrl_holdouts, "feature_seed": int(feature_seed),
            "n_features": int(Xtr.shape[1]), "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
            "n_novel_combo_heldout": int(len(inh_idx)), "n_memctrl_heldout": int(len(memctrl_idx))}
    return (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, {"inh_idx": inh_idx, "memctrl_idx": memctrl_idx}


# ============================================================================================================
# POSITIVE CONTROL task (built-in): XOR-over-a-pooled-recovered-bit -- the part-2 target the arc found GENUINELY
# depth-required. Proves this exact MLP harness CAN detect depth-separation when it is present, so a role-filler
# NEGATIVE is trustworthy (the harness is not blind to depth). Same (Xtr,ytr,Ltr),(Xte,yte,Lte),meta,idx contract.
# ============================================================================================================
def make_task_xor_pool_positive_control(seed, n_bits=4, pool=4, n=4000, noise=0.6, held_frac=0.25):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, n_bits * pool)); y = np.zeros(n, np.int64); L = np.zeros((n, n_bits))
    for i in range(n):
        bits = rng.integers(0, 2, size=n_bits)
        row = []
        for b in range(n_bits):
            base = 1.0 if bits[b] == 1 else -1.0
            row.extend(base + noise * rng.standard_normal(pool))
        X[i] = row; y[i] = int(np.bitwise_xor.reduce(bits)); L[i] = bits
    ntr = int(n * (1 - held_frac))
    mu = X[:ntr].mean(0, keepdims=True); sd = X[:ntr].std(0, keepdims=True)
    X = (X - mu) / (sd + 1e-6)
    Xtr, ytr, Ltr = X[:ntr], y[:ntr], L[:ntr]
    Xte, yte, Lte = X[ntr:], y[ntr:], L[ntr:]
    # the whole held set is the "novel" test for the control (all rows are inferable by the pooled-bit XOR).
    inh_idx = np.arange(len(yte), dtype=np.int64)
    meta = {"control": "xor_over_pool", "n_bits": n_bits, "pool": pool, "k_classes": 2,
            "n_features": int(Xtr.shape[1]), "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
            "n_novel_combo_heldout": int(len(inh_idx)), "n_memctrl_heldout": 0}
    return (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, {"inh_idx": inh_idx, "memctrl_idx": np.array([], np.int64)}


# ============================================================================================================
# Emergence probe (secondary/reported): does a linear read-out of the FROZEN hidden rep recover the recovered
# role+filler KEY bits? (== the intermediate feature the bind must build.)
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


def _linear_probe_class(Xtr, ytr, Xte, yte, idx, n_class):
    """Stage-0 LINEAR PROBE: a ridge one-vs-all linear classifier on the raw CODES. If held-out NOVEL-combo binding
    is linearly decodable, the task is linearly shortcut-able (the concat/score shortcut) and NOT a genuine
    conjunction -> the depth question is moot. Must be near chance for a genuine nonlinear bind."""
    if idx is None or len(idx) == 0:
        return float("nan")
    Xtr1 = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte1 = np.concatenate([Xte[idx], np.ones((len(idx), 1))], 1)
    Y = np.eye(n_class)[ytr]
    lam = 1e-2 * np.eye(Xtr1.shape[1]); lam[-1, -1] = 0.0
    W = np.linalg.solve(Xtr1.T @ Xtr1 + lam, Xtr1.T @ Y)
    pred = np.argmax(Xte1 @ W, 1)
    return float(np.mean(pred == yte[idx]))


def _train_oracle(net, X, y, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode="oracle", lr=lr)


def _acc_on(net, X, y, idx):
    """Held-out accuracy on a SUBSET of the held set (the binding test / the memctrl set), not the whole held set."""
    if idx is None or len(idx) == 0:
        return float("nan")
    _, lg = net._forward(np.asarray(X[idx], float))
    return float(np.mean(np.argmax(np.asarray(lg), 1) == np.asarray(y[idx])))


# ============================================================================================================
# STAGE 0 -- depth-genuineness on the HELD-OUT NOVEL-COMBINATION split (the SELF-CORRECTING gate, MEASURED FIRST).
# ============================================================================================================
def stage0_depth_genuineness(task, idx, k, hidden, epochs, lr, batch, seed):
    """A 1-hidden-layer fenced-backprop oracle must UNDERFIT held-out NOVEL-combination binding while a 2-3-hidden
    oracle clears it by a clear margin AND the LINEAR probe is near chance (not linearly shortcut-able). ALL arms are
    the fenced backprop ORACLE (this measures the representational depth-requirement of the TASK, not the credit
    rule). The gated accuracy is on the novel-combination held-out subset (both role+filler seen in some train combo)
    -- the binding test. Reports 0/1/2/3-hidden oracle held-out + gaps + chance + the linear probe."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    inh_idx = idx["inh_idx"]
    n_in = Xtr.shape[1]
    if len(inh_idx):
        yv = yte[inh_idx]
        chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    lin_probe = _linear_probe_class(Xtr, ytr, Xte, yte, inh_idx, k)

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
    # depth-separating requires: the deep oracle clears >=0.80, a real gap over the best shallow, AND the linear probe
    # is near chance (else it is linearly shortcut-able, not a genuine conjunction).
    lin_near_chance = bool(np.isnan(lin_probe) or lin_probe <= chance + 0.10)
    depth_separating = bool(deep_best >= 0.80 and depth_gap >= 0.05 and deep_best > l1_te + 0.03 and lin_near_chance)
    return {"chance": chance, "n_features": int(n_in), "linear_probe": lin_probe, "lin_near_chance": lin_near_chance,
            "linear_novel_heldout": l0_te, "linear_train": l0_tr,
            "l1_novel_heldout": l1_te, "l1_train": l1_tr,
            "l2_novel_heldout": l2_te, "l2_train": l2_tr,
            "l3_novel_heldout": l3_te, "l3_train": l3_tr,
            "deep_best_novel_heldout": deep_best, "depth_gap": float(depth_gap),
            "depth_separating": depth_separating}


# ============================================================================================================
# STAGE 1 -- the deep-credit arms + per-layer alignment (the GATED metric) + anti-cheats (incl. the Trap-B defeat
# and the MEMORIZATION control).  (Structure mirrors _semantic_inheritance_deep_credit_derisk verbatim.)
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
    return {"novel_heldout": _acc_on(net, Xte, yte, inh_idx), "memctrl_heldout": _acc_on(net, Xte, yte, mem_idx),
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
    return {"novel_heldout": _acc_on(net, Xte, yte, idx["inh_idx"]),
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
    res["oracle"] = {"novel_heldout": _acc_on(onet, Xte, yte, inh_idx),
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
    # 1-hidden floor (memorization/no-depth): must UNDERFIT held-out novel-combination binding.
    fnet = Net(shal, seed=seed, beta=beta, p0=p0)
    _train(fnet, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    res["single_layer"] = {"novel_heldout": _acc_on(fnet, Xte, yte, inh_idx),
                           "memctrl_heldout": _acc_on(fnet, Xte, yte, mem_idx), "train": float(fnet.accuracy(Xtr, ytr))}

    # --- anti-cheats ---
    res["wrong_sign"] = _wrongsign_alignment(Net, deep, task, idx, epochs, lr, batch, seed, _kind, beta=beta, p0=p0)
    res["wrong_sign_plain_fa"] = _wrongsign_alignment(FANet, deep, task, idx, epochs, lr, batch, seed, "fa",
                                                      beta=beta, p0=p0)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    pnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(pnet, Xtr, yperm, "bdsp", epochs, lr, batch, seed)
    res["permuted"] = {"novel_heldout": _acc_on(pnet, Xte, yte, inh_idx), "train": float(pnet.accuracy(Xtr, yperm))}
    lnet = Net(deep, seed=seed, beta=beta, p0=p0)
    _train(lnet, Xtr, ytr, "apical_lesion", epochs, lr, batch, seed)
    ab = Xtr[:min(len(Xtr), 512)]; aby = ytr[:min(len(ytr), 512)]
    l_align = _per_layer_alignment(lnet, ab, aby, _kind)
    res["apical_lesion"] = {"novel_heldout": _acc_on(lnet, Xte, yte, inh_idx), "train": float(lnet.accuracy(Xtr, ytr)),
                            "per_layer_alignment": [float(c) for c in l_align],
                            "deepest_layer_alignment": float(l_align[0])}

    b0 = Net(deep, seed=seed, beta=beta, p0=p0); f0 = DendriticMLP(deep, seed=seed)
    res["same_init_as_oracle"] = bool(all(np.allclose(a, b) for a, b in zip(b0.W, f0.W)))
    return res


def run_seed(seed, k, hidden, epochs, lr, batch, rule, feedback, homeostasis, kp_lr, kp_decay, beta, p0,
             task_kwargs, deep_layers=2, control=False):
    if control:
        task_full = make_task_xor_pool_positive_control(seed)
    else:
        task_full = make_task_rolefiller_binding(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    if k is None:
        k = meta["k_classes"]
    s0 = stage0_depth_genuineness(task, idx, k, hidden, epochs, lr, batch, seed)
    out = {"seed": seed, "meta": meta, "stage0_depth_genuineness": s0}
    # Stage 1 runs ONLY when Stage 0 is depth-separating (the self-correcting gate). Otherwise reading the credit
    # arms is meaningless (the arc's discipline: gate FIRST, do not read the arms on a non-depth-separating task).
    if s0["depth_separating"]:
        out["stage1_deep_credit"] = stage1_deep_credit(task, idx, k, hidden, epochs, lr, batch, seed, rule=rule,
                                                        feedback=feedback, homeostasis=homeostasis, kp_lr=kp_lr,
                                                        kp_decay=kp_decay, beta=beta, p0=p0, deep_layers=deep_layers)
    else:
        out["stage1_deep_credit"] = None
    return out


def _fmt_align(a):
    return "[" + ", ".join(f"{c:.2f}" for c in a) + "]"


def main():
    ap = argparse.ArgumentParser(description="Role-filler binding / systematic-recombination deep-credit de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=300)
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
    ap.add_argument("--deep-layers", type=int, default=2)
    ap.add_argument("--n-roles", type=int, default=14)
    ap.add_argument("--n-fillers", type=int, default=14)
    ap.add_argument("--key-bits", type=int, default=2, help="recoverable key bits per role/filler -> 2^key_bits classes")
    ap.add_argument("--sub-k", type=int, default=2, help="code dims per key bit (sign-parity recovery = hop 1)")
    ap.add_argument("--n-obs", type=int, default=18, help="observations per TRAIN combo (noisy realizations)")
    ap.add_argument("--held-frac", type=float, default=0.30, help="fraction of combos held out as NOVEL combinations")
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--id-dim", type=int, default=3)
    ap.add_argument("--bind-op", choices=["xor", "and", "mux"], default="xor",
                    help="the role x filler conjunction gate (the hop-2 bind)")
    ap.add_argument("--memctrl-holdouts", type=int, default=2,
                    help="roles held out of ALL training (their held combos = the un-inferable memorization control)")
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--positive-control", action="store_true",
                    help="run the XOR-over-pool POSITIVE CONTROL (proves the harness CAN detect depth when present)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_roles=a.n_roles, n_fillers=a.n_fillers, key_bits=a.key_bits, sub_k=a.sub_k, n_obs=a.n_obs,
                       held_frac=a.held_frac, noise=a.noise, id_dim=a.id_dim, bind_op=a.bind_op,
                       memctrl_holdouts=a.memctrl_holdouts, feature_seed=a.feature_seed)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, None, a.hidden, a.epochs, a.lr, a.batch, a.rule, a.feedback, a.homeostasis,
                         a.kp_lr, a.kp_decay, a.beta, a.p0, task_kwargs, deep_layers=a.deep_layers,
                         control=a.positive_control)
            per.append(r)
            s0 = r["stage0_depth_genuineness"]; s1 = r["stage1_deep_credit"]; m = r["meta"]
            print("-" * 112, flush=True)
            tag = "POSITIVE-CONTROL xor-over-pool" if a.positive_control else \
                  (f"{m['n_roles']}x{m['n_fillers']} roles/fillers | bind={m['bind_op']} | "
                   f"{m['k_classes']} classes | key_bits={m['key_bits']} sub_k={m['sub_k']}")
            print(f"[seed {s}] {tag} | {m['n_features']} feats | {m['n_train']} train / "
                  f"{m['n_novel_combo_heldout']} novel-held / {m['n_memctrl_heldout']} memctrl-held | "
                  f"chance {s0['chance']:.3f}", flush=True)
            print(f"  STAGE0 depth-genuineness (held-out NOVEL-combination acc): linear {s0['linear_novel_heldout']:.3f} | "
                  f"1-layer {s0['l1_novel_heldout']:.3f} | 2-layer {s0['l2_novel_heldout']:.3f} | "
                  f"3-layer {s0['l3_novel_heldout']:.3f} | deep-best {s0['deep_best_novel_heldout']:.3f} | "
                  f"depth-gap {s0['depth_gap']:+.3f} | lin-probe {s0['linear_probe']:.3f} (near-chance "
                  f"{s0['lin_near_chance']}) => DEPTH-SEPARATING {s0['depth_separating']}", flush=True)
            if s1 is None:
                print("  STAGE1 SKIPPED -- Stage 0 not depth-separating (the self-correcting gate; reading the deep-"
                      "credit arms on a non-depth-separating task would be meaningless).", flush=True)
                continue
            tf = s1["test_fixed"]; tl = s1["test_learned"]; th = s1["test_learned_homeo"]; pf = s1["plain_fa"]
            ws = s1["wrong_sign"]; les = s1["apical_lesion"]
            print(f"  STAGE1 [{s1['rule']}] per-layer align vs oracle (layer0=deepest) + held-out NOVEL-combo acc:",
                  flush=True)
            print(f"    test-fixed   novel {tf['novel_heldout']:.3f} memctrl {tf['memctrl_heldout']:.3f} "
                  f"align {_fmt_align(tf['per_layer_alignment'])} deep {tf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    test-learned novel {tl['novel_heldout']:.3f} align {_fmt_align(tl['per_layer_alignment'])} "
                  f"deep {tl['deepest_layer_alignment']:.2f} (KP, transport-free {tl['no_weight_transport']})", flush=True)
            print(f"    +homeo       novel {th['novel_heldout']:.3f} align {_fmt_align(th['per_layer_alignment'])} "
                  f"deep {th['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    plain-FA     novel {pf['novel_heldout']:.3f} align {_fmt_align(pf['per_layer_alignment'])} "
                  f"deep {pf['deepest_layer_alignment']:.2f}", flush=True)
            print(f"    single-layer novel {s1['single_layer']['novel_heldout']:.3f} | "
                  f"oracle-deep novel {s1['oracle']['novel_heldout']:.3f} | chance {s1['chance']:.3f}", flush=True)
            print(f"    [anti-cheat] WRONG-SIGN: novel {ws['novel_heldout']:.3f} (may be rescued) | ALIGNMENT deep "
                  f"{ws['deepest_layer_alignment']:.2f} align {_fmt_align(ws['per_layer_alignment'])}  <- must FAIL", flush=True)
            print(f"    [anti-cheat] permuted {s1['permuted']['novel_heldout']:.3f} (~chance) | lesion novel "
                  f"{les['novel_heldout']:.3f} align deep {les['deepest_layer_alignment']:.2f} | "
                  f"MEMCTRL(oracle) {s1['oracle']['memctrl_heldout']:.3f} (must ~chance) | "
                  f"same-init {s1['same_init_as_oracle']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "rolefiller_binding_deep_credit", "seeds": a.seeds, "rule": a.rule,
               "positive_control": bool(a.positive_control),
               "config": {"hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                          "feedback": a.feedback, "homeostasis": bool(a.homeostasis), "kp_lr": a.kp_lr,
                          "kp_decay": a.kp_decay, "beta": a.beta, "p0": a.p0, "task": task_kwargs,
                          "deep_layers": a.deep_layers, "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(path, default=float("nan")):
            out = []
            for p in per:
                v = p
                ok = True
                for kk in path:
                    if v is None or (isinstance(v, dict) and kk not in v):
                        ok = False; break
                    v = v[kk]
                if ok and v is not None:
                    out.append(v)
            return float(np.nanmean(out)) if out else default

        s0_sep = all(p["stage0_depth_genuineness"]["depth_separating"] for p in per)
        deep_best = _m(["stage0_depth_genuineness", "deep_best_novel_heldout"])
        l1 = _m(["stage0_depth_genuineness", "l1_novel_heldout"]); depth_gap = _m(["stage0_depth_genuineness", "depth_gap"])
        lin_probe = _m(["stage0_depth_genuineness", "linear_probe"])
        ch = _m(["stage0_depth_genuineness", "chance"])
        summary["stage0_depth_separating"] = s0_sep
        summary["aggregate_stage0"] = {"deep_best_novel_heldout": deep_best, "l1_novel_heldout": l1,
                                       "depth_gap": depth_gap, "linear_probe": lin_probe, "chance": ch}
        if not s0_sep:
            # THE SELF-CORRECTING GATE FIRED -- honest boundary, Stage 1 correctly did NOT run.
            summary["SIGNAL"] = False
            summary["verdict"] = (
                f"STAGE-0 BOUNDARY (honest) -- role-filler binding as a SUPERVISED CLASSIFICATION of a held-out NOVEL "
                f"(role, filler) combination is NOT depth-separating at n_roles={a.n_roles}/n_fillers={a.n_fillers}/"
                f"bind={a.bind_op}/key_bits={a.key_bits}/sub_k={a.sub_k}/noise={a.noise} (deep-best novel {deep_best:.3f} "
                f"vs 1-layer {l1:.3f}, gap {depth_gap:+.3f}; linear-probe {lin_probe:.3f} vs chance {ch:.3f}). A SINGLE "
                f"wide hidden layer forms the role x filler conjunctive feature directly (universal approximation over "
                f"the disjoint code blocks) and generalizes it to novel combos, so adding depth does not help (often "
                f"hurts by overfitting the arbitrary id codes). The linear probe IS near/below chance (the bind is a "
                f"genuine nonlinear conjunction, NOT the concat/score shortcut) -- but that nonlinearity is a "
                f"1-hidden-layer nonlinearity. ARC READ: the deep-credit depth-benefit is NARROWER still than 'nonlinear "
                f"conjunction' -- a SINGLE role-filler bind is depth-1; the depth lives in SEQUENTIAL/RECURSIVE "
                f"composition (a multi-hop bind: bind then re-bind / role-filler chains), which is the honest next "
                f"instrument. Run --positive-control to confirm the harness CAN detect depth (the part-2 XOR-over-pool "
                f"IS depth-separating on this exact harness) => this negative is trustworthy, not a blind harness. "
                f"The self-correcting Stage-0 gate did its job: it did NOT read the deep-credit arms on a "
                f"non-depth-separating task.")
        else:
            oracle = _m(["stage1_deep_credit", "oracle", "novel_heldout"])
            oracle_mem = _m(["stage1_deep_credit", "oracle", "memctrl_heldout"])
            tf_deep = _m(["stage1_deep_credit", "test_fixed", "deepest_layer_alignment"])
            tl_deep = _m(["stage1_deep_credit", "test_learned", "deepest_layer_alignment"])
            th_deep = _m(["stage1_deep_credit", "test_learned_homeo", "deepest_layer_alignment"])
            pf_deep = _m(["stage1_deep_credit", "plain_fa", "deepest_layer_alignment"])
            best_test_deep = max(tf_deep, tl_deep, th_deep)
            tf_inh = _m(["stage1_deep_credit", "test_fixed", "novel_heldout"])
            tl_inh = _m(["stage1_deep_credit", "test_learned", "novel_heldout"])
            th_inh = _m(["stage1_deep_credit", "test_learned_homeo", "novel_heldout"])
            best_test_inh = max(tf_inh, tl_inh, th_inh)
            sl_inh = _m(["stage1_deep_credit", "single_layer", "novel_heldout"])
            ws_deep = _m(["stage1_deep_credit", "wrong_sign", "deepest_layer_alignment"])
            les_deep = _m(["stage1_deep_credit", "apical_lesion", "deepest_layer_alignment"])
            les_inh = _m(["stage1_deep_credit", "apical_lesion", "novel_heldout"])
            perm = _m(["stage1_deep_credit", "permuted", "novel_heldout"])
            wt = all(p["stage1_deep_credit"]["test_learned"]["no_weight_transport"]
                     and p["stage1_deep_credit"]["same_init_as_oracle"] for p in per)
            wrongsign_fails = bool(ws_deep < best_test_deep - 0.10 and ws_deep < 0.30)
            lesion_collapses = bool(les_deep < best_test_deep - 0.10)
            permuted_chance = bool(perm <= ch + 0.08)
            align_signal = bool(best_test_deep > pf_deep - 0.02)
            learns_bind = bool(best_test_inh > sl_inh + 0.05 and best_test_inh > ch + 0.05)
            memctrl_holds = bool(np.isnan(oracle_mem) or oracle_mem <= ch + 0.15)
            oracle_ok = bool(oracle >= 0.80)
            signal = bool(s0_sep and oracle_ok and best_test_deep > 0.15 and learns_bind and wrongsign_fails
                          and lesion_collapses and permuted_chance and wt and memctrl_holds)
            summary["aggregate_stage1"] = {
                "oracle_novel_heldout": oracle, "oracle_memctrl_heldout": oracle_mem,
                "deepest_align_test_fixed": tf_deep, "deepest_align_test_learned": tl_deep,
                "deepest_align_test_learned_homeo": th_deep, "deepest_align_plain_fa": pf_deep,
                "best_test_deep_align": best_test_deep, "novel_test_fixed": tf_inh, "novel_test_learned": tl_inh,
                "novel_test_learned_homeo": th_inh, "best_test_novel": best_test_inh, "single_layer_novel": sl_inh,
                "learns_bind": learns_bind, "wrong_sign_deep_align": ws_deep,
                "wrong_sign_fails_alignment": wrongsign_fails, "lesion_deep_align": les_deep,
                "lesion_collapses": lesion_collapses, "permuted_novel": perm, "permuted_chance": permuted_chance,
                "memctrl_holds": memctrl_holds, "no_weight_transport": wt, "chance": ch}
            summary["SIGNAL"] = signal
            if not oracle_ok:
                summary["verdict"] = (f"INCONCLUSIVE -- the depth oracle only reached {oracle:.3f} held-out novel-combo "
                                      f"at H{a.hidden}; tune epochs/lr/hidden before reading the arms (NOT a verdict).")
            else:
                _tb = "FAILS (deep align {:.2f} < best-test {:.2f})".format(ws_deep, best_test_deep) if wrongsign_fails \
                      else "does NOT fail (deep align {:.2f}) -- Trap B may re-bite; report honestly".format(ws_deep)
                _bind = "LEARNS the bind" if learns_bind else "does NOT beat the 1-layer floor"
                _mem = "holds" if memctrl_holds else "LEAKS"
                _head = ("LOAD-BEARING depth-benefit on ROLE-FILLER BINDING (the culminating goal-relevant capability)"
                         if signal else "NO clean deep-credit signal yet (see the arm table)")
                _next = ("controller runs 6-seed + adversarial-verify + on-bridge spiking"
                         if signal else "honest read: escalate / diagnose")
                summary["verdict"] = (
                    f"STAGE-0 PASS (depth-separating: deep-best novel {deep_best:.3f} vs 1-layer {l1:.3f}, gap "
                    f"{depth_gap:+.3f}, oracle {oracle:.3f}, lin-probe {lin_probe:.3f} << chance {ch:.3f}). STAGE-1 deep "
                    f"credit ({a.rule}): held-out NOVEL-combo binding -- single-layer {sl_inh:.3f}, best-test "
                    f"{best_test_inh:.3f}, chance {ch:.3f} ({_bind}); deepest-layer alignment -- plain-FA {pf_deep:.2f}, "
                    f"best-test {best_test_deep:.2f}; lesion novel {les_inh:.3f} align {les_deep:.2f}; WRONG-SIGN "
                    f"alignment {_tb}; permuted {perm:.3f} (~chance {ch:.3f}); memorization-control(oracle) "
                    f"{oracle_mem:.3f} ({_mem}); no weight transport {wt}. {_head} => {_next}. Numpy RATE reference; "
                    f"the decisive on-bridge spiking depth-3 multi-seed is the controller's GPU run.")
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[rolefiller-binding-deep-credit] {summary['verdict']}", flush=True)
    print(f"[rolefiller-binding-deep-credit] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
