"""FHRR-B Option 3 — orthogonal per-attribute role tags (tensor-product representation): does a DIFFERENT
FIXED structure make the multi-attribute bundle-inverse RECOVERABLE + GENERALIZING without the exact reciprocal?

CONTEXT (verify against the scoping `2026-06-20-FHRR-B-learned-binder-scoping.md`, Option-1 GO, Option-2 NEGATIVE).
The production composer binds `role (x) filler` and BUNDLES a fact (= superposes the three role-filler bindings of
an SVO sentence into one vector), then recovers one role's filler from that superposition by a FIXED, exactly-
invertible FHRR self-inverse algebra (the residual host-DESIGNED shortcut "FHRR-B"). Throughout:
  - "bundle" = superpose multiple role-filler bindings into one fact vector;
  - "unbind from a bundle" = recover one role's filler from that sum;
  - "(x)" = elementwise (Hadamard) product;
  - "tensor-product representation (TPR, Smolensky 1990)" = bind a role and a filler by their OUTER product so
    that DISTINCT roles occupy DISJOINT / ORTHOGONAL subspaces of the fact vector -- the bindings then do NOT
    interfere (no superposition crosstalk within a role's subspace), so recovering one role's filler is a fixed
    ORTHOGONAL PROJECTION onto that role's subspace, NOT an exact multiplicative reciprocal.

WHY OPTION 3 (the precise question). Option 2 proved the BIND FORM (the role-dependent reciprocal that recovers a
filler from a 3-way superposition) is NOT learnable even by a DEEP / hidden-layer binder, because the missing
operation is an EXACT element-wise reciprocal a learned projection cannot satisfy
(`2026-06-20-FHRR-B-option2-deep-binder-derisk.md`: single-binding held-out 0.000 deep-concat / 0.250 deep-gated
vs 1.000 fixed self-inverse). Option 3 SIDESTEPS the reciprocal entirely: if attributes get DISTINCT, ORTHOGONAL
role tags (a tensor-product structure), the bindings are SEPARABLE -- each lives in its own subspace -- so NO
exact-inverse is needed: recover attribute r by a fixed orthogonal projection onto role-r's subspace + a LEARNED
cleanup (read-out). The decisive test (Fodor-Pylyshyn systematicity): does this recover HELD-OUT (never-bundled)
role-filler combinations, leakage-asserted -- NOT raw recall.

HONEST FRAMING (stated in the doc too). Orthogonal-TPR role tags are themselves a FIXED structural choice (an
outer-product / a fixed block assignment), NOT a learned bind. So Option 3 tests whether a DIFFERENT FIXED
structure makes multi-attribute recovery LEARNABLE + GENERALIZING (fillers + the decomposition cleanup are
learned; the orthogonal role-subspace structure is fixed), NOT whether the BIND ITSELF is learned. If Option 3
GOes, FHRR-B's multi-attribute capability closes via learned codes + a separable structure + a learned read-out.
If Option 3 ALSO fails to generalize, the evidence converges (Option-1 cleanup GO + Option-2/3 bind-learning
NEGATIVE) on a genuine finding: the role-filler BIND is a FIXED STRUCTURAL neural primitive
(binding-by-coincidence / dendritic multiplication), not a learnable host op -- present to controller, NOT a
"closed boundary".

THE MECHANISM (this de-risk).
  bind:  each role r is assigned a DISJOINT block of width d_role in the fact vector. The filler is projected by
         a LEARNED shared map  w = filler @ W_F  (w in R^{d_role}), and PLACED into role-r's block. (This is the
         block-diagonal / orthogonal-role-tag form of a tensor product: role tag = a fixed one-hot-over-blocks =
         an orthonormal role basis; the outer product role (x) filler lands ONLY in role-r's block.) The fact
         vector  T = concat_r [ block_r ]  with block_r = w_r placed in slot r (other slots zero for that
         binding) -> bundle T = sum_r block_r is SEPARABLE: block r holds ONLY role-r's filler projection, with
         ZERO crosstalk from other roles.
  unbind (recover role r's filler): a FIXED orthogonal projection reads block r out of T (= slice block r), then
         a LEARNED cleanup MLP maps that block -> a filler estimate -> nearest filler. NO reciprocal: the role
         selects WHICH orthogonal subspace to read; the learned cleanup decodes the filler.
  The filler projection W_F and the cleanup are SHARED across all role blocks (one decoder, applied per block) --
  this is what forces SYSTEMATICITY: a filler seen in block-a's training must be decodable from block-i at test.
  All trained bundle-aware by backprop (a host-shortcut CEILING characterization per the scoping: a PASS here =
  "a spiking read-out of this separable form CAN be systematic", the gate before any spiking realization;
  explicitly NOT "the brain binds").

ABLATIONS (the two scoping-mandated levers):
  - attribute count A in {2,3,4} (capacity: more superposed bindings -> harder, but TPR has no crosstalk so it
    should hold where the additive ±1 bundle degrades).
  - role-subspace dim d_role in {32,64,128} (how much room each orthogonal block gives the filler projection +
    the cleanup).

ANTI-CHEATS (full battery -- mirrors the established 0.989-vs-0.193 contrast):
  1. The prior NEGATIVE controls still fail on the identical corpus/splits/seeds (the shallow learned-linear
     0.056 / additive 0.193 / dendritic 0.168 references; re-run a live additive-bundle learned control here).
  2. FIXED-±1 self-inverse POSITIVE control carries (~0.989) -- proves the harness detects a working bundling
     bind, so any NEGATIVE is real, not a broken eval.
  3. HELD-OUT systematicity, leakage-asserted (make_systematicity_splits), vs the memorization floor + chance
     (1/F). The bar is HELD-OUT generalization, NEVER raw recall.
  4. MEMORIZATION-FLOOR control: shuffle the role->filler labels at train time -> held-out collapses to chance.
  5. PERMUTED-ROLE control: read the WRONG block (query the wrong role) -> collapses to chance (proves the
     orthogonal separation + role-keyed read-out is load-bearing).
  6. LESION control: scramble the learned cleanup -> held-out collapses (the cleanup is load-bearing; the fixed
     projection ALONE is not the answer).
  7. THE MOAT (no-confab familiarity gap): conf on a real bound filler >> on a novel OOD filler -- reported
     every seed, NEVER weakened.
  8. DECORRELATED stream codes (primary; the code-correlation wall is a SEPARATE, already-solved axis). An
     optional correlated-codes pass is wired (--codes neural).

Reuse-by-import (make_role_codes / make_systematicity_splits / native_argmax); cached 320 stream codes; CPU;
numpy-only; NO GPU; NO sim/ edits.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_fhrr_b_option3_orthogonal_tpr_derisk \
          --seeds 42,43,44,100,101,102 --attrs 2,3,4 --d-role 32,64,128 --run-anticheats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)

R_MAX, F, N_SPLITS = 4, 16, 3
N_FACT_STEPS = 24000        # bundle-aware training steps (matches the Option-1/2 de-risks)
N_EVAL_FACTS = 40
LR = 0.004
LAM = 1e-4

# Established reference numbers on this harness (cited; the additive arm re-runs a live control)
SHALLOW_ADDITIVE = 0.193
SHALLOW_LEARNED_LINEAR = 0.056
SHALLOW_DENDRITIC = 0.168
OPTION1_CLEANUP = 1.000      # Option-1 learned iterative cleanup over the FIXED bind @ D_h=256
FIXED_FHRR_CEILING = 0.989
DEEP_BINDER_NEG = None       # filled from Option-2 doc table if available (gated d=2 ~ shallow)


def _glorot(rng, shape):
    return rng.standard_normal(shape) * (1.0 / np.sqrt(shape[0]))


def _unit(x):
    return x / (np.linalg.norm(x) + 1e-9)


# ============================================================================
# OPTION 3 — ORTHOGONAL-TPR binder with a learned (per-block, shared) cleanup
# ============================================================================

class OrthogonalTPRBinder:
    """Tensor-product (orthogonal-role-tag) binder.

    Each of the A roles gets a DISJOINT block of width d_role in the fact vector (the orthogonal role basis = a
    one-hot-over-blocks role tag).  A filler is projected by a LEARNED shared map  w = filler @ W_F  (R^{d_role})
    and PLACED into role-r's block; the fact T = concat over blocks; the bundle = the sum (block r holds ONLY
    role-r's filler projection -> NO crosstalk, the whole point of the orthogonal subspaces).

    Recover role r's filler: read block r (a FIXED orthogonal projection = slice block r) -> a LEARNED cleanup
    MLP -> filler estimate.  W_F + the cleanup are SHARED across blocks (one decoder applied per block), which is
    what forces systematicity (a filler seen in one block must decode from any block).

    A == n_roles (attribute count).  n_hidden tanh hidden layers in the cleanup (width d_h, default = d_role*2).
    """

    def __init__(self, D_in, A, d_role=64, d_h=None, n_hidden=1, lr=LR, lam=LAM, seed=42):
        self.D_in, self.A, self.d_role = D_in, int(A), int(d_role)
        self.d_h = int(d_h) if d_h else max(2 * d_role, 64)
        self.n_hidden = int(n_hidden)
        self.lr, self.lam = lr, lam
        rng = np.random.default_rng(seed * 17 + 3)

        # Learned SHARED filler projection into a role block.
        self.W_F = _glorot(rng, (D_in, self.d_role))
        self.params = ["W_F"]

        # The SHARED learned cleanup MLP: block (R^{d_role}) -> filler estimate (R^{D_in}).
        self.C_W, self.C_b = [], []
        in_dim = self.d_role
        for _ in range(self.n_hidden):
            self.C_W.append(_glorot(rng, (in_dim, self.d_h)))
            self.C_b.append(np.zeros(self.d_h))
            in_dim = self.d_h
        self.C_Wout = _glorot(rng, (in_dim, D_in))
        self.C_bout = np.zeros(D_in)
        for i in range(self.n_hidden):
            self.params += [f"C_W{i}", f"C_b{i}"]
        self.params += ["C_Wout", "C_bout"]

        self.t = 0
        self._m = {p: np.zeros_like(self._get(p)) for p in self.params}
        self._v = {p: np.zeros_like(self._get(p)) for p in self.params}

    def _get(self, name):
        if name.startswith("C_W") and name[3:].isdigit():
            return self.C_W[int(name[3:])]
        if name.startswith("C_b") and name[3:].isdigit():
            return self.C_b[int(name[3:])]
        return getattr(self, name)

    def _adam(self, name, grad):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = self._m[name], self._v[name]
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * grad * grad
        mhat = m / (1 - b1 ** self.t); vhat = v / (1 - b2 ** self.t)
        self._get(name)[...] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    # ---------------- forward ----------------
    def bind_block(self, filler):
        """The filler's projection into a role block (R^{d_role})."""
        return filler @ self.W_F

    def bundle(self, fillers_by_role):
        """fillers_by_role: list of A filler vectors (one per role) -> fact vector (R^{A*d_role})."""
        return np.concatenate([self.bind_block(f) for f in fillers_by_role])

    def read_block(self, T, r):
        """FIXED orthogonal projection: read role-r's block out of the fact vector."""
        return T[r * self.d_role:(r + 1) * self.d_role]

    def _cleanup_forward(self, blk, cache=False):
        acts = [blk]; pre = []
        h = blk
        for i in range(self.n_hidden):
            z = h @ self.C_W[i] + self.C_b[i]
            pre.append(z); h = np.tanh(z); acts.append(h)
        est = h @ self.C_Wout + self.C_bout
        return (est, {"acts": acts, "pre": pre}) if cache else est

    def unbind(self, T, r):
        return self._cleanup_forward(self.read_block(T, r), cache=False)

    # ---------------- training (bundle-aware backprop) ----------------
    def train_fact_step(self, fillerids, fillers, query_r, target_fid):
        self.t += 1
        # Forward: bundle the A bindings, read the queried block, clean it up.
        w_list = [fillers[f] @ self.W_F for f in fillerids]   # per-role filler projection
        blk = w_list[query_r]                                 # block of the queried role (orthogonal -> isolated)
        est, mc = self._cleanup_forward(blk, cache=True)
        err = est - fillers[target_fid]
        loss = float(np.mean(err ** 2))

        # ---- backward through the cleanup MLP ----
        d_est = 2.0 * err / self.D_in
        acts, pre = mc["acts"], mc["pre"]
        h_last = acts[-1]
        grads = {"C_Wout": np.outer(h_last, d_est) + self.lam * self.C_Wout, "C_bout": d_est.copy()}
        dh = self.C_Wout @ d_est
        for i in reversed(range(self.n_hidden)):
            dz = dh * (1.0 - np.tanh(pre[i]) ** 2)
            grads[f"C_W{i}"] = np.outer(acts[i], dz) + self.lam * self.C_W[i]
            grads[f"C_b{i}"] = dz
            dh = self.C_W[i] @ dz
        # dh = grad wrt blk (the queried role's filler projection)

        # ---- backward through W_F (only the queried role's filler contributed to blk) ----
        # NOTE: orthogonality means ONLY the queried block carries gradient -> the other roles' fillers in the
        # bundle do not affect this query's loss (zero crosstalk; the structural win + the systematicity test).
        grads["W_F"] = np.outer(fillers[fillerids[query_r]], dh) + self.lam * self.W_F

        for name in self.params:
            self._adam(name, grads[name])
        return loss


# ============================================================================
# Per-seed run for one (attribute count A, d_role) cell
# ============================================================================

def run_seed_cell(codes, seed, A, d_role, n_hidden=1, shuffle_train=False, lesion=False):
    splits = make_systematicity_splits(R_MAX, F, N_SPLITS, seed)
    fillers = codes[:F]
    D_in = fillers.shape[1]
    rng = np.random.default_rng(seed * 53 + 9)

    single_held, bundle_train, bundle_held, perm_role_held = [], [], [], []
    moat_known, moat_novel = [], []

    for split in splits:
        # The split is over R_MAX=4 roles; for A<=R_MAX we use the first A roles' train/held-out pairs.
        train_set = set((r, f) for (r, f) in split["train"] if r < A)
        held_set = set((r, f) for (r, f) in split["held_out"] if r < A)
        binder = OrthogonalTPRBinder(D_in=D_in, A=A, d_role=d_role, n_hidden=n_hidden, lr=LR, lam=LAM, seed=seed)
        tr_by_role = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(A)}
        if min(len(tr_by_role[r]) for r in range(A)) == 0:
            continue
        for _ in range(N_FACT_STEPS):
            fids = [int(rng.choice(tr_by_role[r])) for r in range(A)]
            qr = int(rng.integers(A))
            tgt = int(rng.integers(F)) if shuffle_train else fids[qr]
            binder.train_fact_step(fids, fillers, qr, tgt)

        if lesion:  # scramble the learned cleanup output weights (load-bearing test)
            lrng = np.random.default_rng(seed * 7 + 1)
            binder.C_Wout = lrng.permutation(binder.C_Wout.reshape(-1)).reshape(binder.C_Wout.shape)

        # --- single-binding held-out (the easiest case: just role r's own block, no other bindings) ---
        sc_n = sc_ok = 0
        for r, f in held_set:
            T = binder.bundle([fillers[f] if rr == r else np.zeros(D_in) for rr in range(A)])
            sc_ok += int(native_argmax(binder.unbind(T, r), fillers) == f); sc_n += 1
        single_held.append(sc_ok / sc_n if sc_n else 0.0)

        # --- BUNDLED held-out (the real test: all A bindings superposed) ---
        ntr_ok = ntr = nh_ok = nh = perm_ok = perm_n = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, A, replace=False)
            T = binder.bundle([fillers[fids[r]] for r in range(A)])
            for r in range(A):
                ok = int(native_argmax(binder.unbind(T, r), fillers) == fids[r])
                if (r, int(fids[r])) in train_set:
                    ntr_ok += ok; ntr += 1
                else:
                    nh_ok += ok; nh += 1
                wrong_r = (r + 1) % A   # permuted-role: read the WRONG block
                perm_ok += int(native_argmax(binder.unbind(T, wrong_r), fillers) == fids[r]); perm_n += 1
        bundle_train.append(ntr_ok / ntr if ntr else 0.0)
        bundle_held.append(nh_ok / nh if nh else 0.0)
        perm_role_held.append(perm_ok / perm_n if perm_n else 0.0)

        # --- the moat (no-confab familiarity gap): real bound filler vs a novel OOD filler ---
        mrng = np.random.default_rng(seed * 99 + split["split_id"] + 1)
        kc, nc = [], []
        for r, f in list(train_set)[:12]:
            T = binder.bundle([fillers[f] if rr == r else np.zeros(D_in) for rr in range(A)])
            est = _unit(binder.unbind(T, r))
            kc.append(float(np.max(fillers @ est)))
        for _ in range(12):
            nov = _unit(mrng.standard_normal(D_in)); r = int(mrng.integers(A))
            T = binder.bundle([nov if rr == r else np.zeros(D_in) for rr in range(A)])
            est = _unit(binder.unbind(T, r))
            nc.append(float(np.max(fillers @ est)))
        moat_known.append(float(np.mean(kc)) if kc else 0.0)
        moat_novel.append(float(np.mean(nc)) if nc else 0.0)

    def m(x):
        return float(np.mean(x)) if x else 0.0
    return {
        "seed": seed, "A": A, "d_role": d_role, "n_hidden": n_hidden,
        "shuffle_train": bool(shuffle_train), "lesion": bool(lesion),
        "single_held": m(single_held), "bundle_train": m(bundle_train), "bundle_held": m(bundle_held),
        "perm_role_held": m(perm_role_held), "moat_known": m(moat_known), "moat_novel": m(moat_novel),
        "moat_gap": m(moat_known) - m(moat_novel),
    }


# ============================================================================
# Live LEARNED-ADDITIVE control (the prior shallow NEGATIVE re-run on the same codes/splits)
# ============================================================================

class _AdditiveLearnedBinder:
    """The shallow learned-ADDITIVE bind + learned-linear unbind that scored ~0.193 — re-run live as a control.
    bind: g = role@W_R + filler@W_F (ADDITIVE, no inverse); bundle = sum; unbind: linear map of [bundle, role]."""

    def __init__(self, D_in, d_h=256, lr=LR, lam=LAM, seed=42):
        rng = np.random.default_rng(seed * 17 + 3)
        self.D_in, self.d_h = D_in, d_h
        self.W_R = _glorot(rng, (D_in, d_h)); self.W_F = _glorot(rng, (D_in, d_h))
        self.W_RU = _glorot(rng, (D_in, d_h)); self.W_U = _glorot(rng, (2 * d_h, D_in)); self.b = np.zeros(D_in)
        self.lr, self.lam = lr, lam; self.t = 0
        self.params = ["W_R", "W_F", "W_RU", "W_U", "b"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self.params}

    def _adam(self, name, grad):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = self._m[name], self._v[name]
        m[:] = b1 * m + (1 - b1) * grad; v[:] = b2 * v + (1 - b2) * grad * grad
        mhat = m / (1 - b1 ** self.t); vhat = v / (1 - b2 ** self.t)
        getattr(self, name)[...] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def bind(self, role, filler):
        return role @ self.W_R + filler @ self.W_F

    def unbind(self, bundle, role):
        return np.concatenate([_unit(bundle), role @ self.W_RU]) @ self.W_U + self.b

    def train_fact_step(self, roleids, fids, roles, fillers, qt, tgt):
        self.t += 1
        gl = [roles[r] @ self.W_R + fillers[f] @ self.W_F for r, f in zip(roleids, fids)]
        bundle = sum(gl); rq = roleids[qt]
        bn = _unit(bundle); ru = roles[rq] @ self.W_RU
        x0 = np.concatenate([bn, ru]); est = x0 @ self.W_U + self.b
        err = est - fillers[tgt]; loss = float(np.mean(err ** 2))
        d_est = 2.0 * err / self.D_in
        d_W_U = np.outer(x0, d_est) + self.lam * self.W_U; d_b = d_est.copy()
        dx0 = self.W_U @ d_est; d_bn = dx0[:self.d_h]; d_ru = dx0[self.d_h:]
        d_W_RU = np.outer(roles[rq], d_ru) + self.lam * self.W_RU
        bnorm = np.linalg.norm(bundle) + 1e-9
        d_bundle = (d_bn - bn * (bn @ d_bn)) / bnorm
        d_W_R = np.zeros_like(self.W_R); d_W_F = np.zeros_like(self.W_F)
        for r, f in zip(roleids, fids):
            d_W_R += np.outer(roles[r], d_bundle); d_W_F += np.outer(fillers[f], d_bundle)
        grads = {"W_R": d_W_R + self.lam * self.W_R, "W_F": d_W_F + self.lam * self.W_F,
                 "W_RU": d_W_RU, "W_U": d_W_U, "b": d_b}
        for nm in self.params:
            self._adam(nm, grads[nm])
        return loss


def run_additive_control(codes, seed, d_h=256):
    splits = make_systematicity_splits(R_MAX, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R_MAX, D_in, seed)
    rng = np.random.default_rng(seed * 53 + 9)
    held = []
    for split in splits:
        train_set = set(split["train"])
        b = _AdditiveLearnedBinder(D_in=D_in, d_h=d_h, seed=seed)
        tr = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
        if min(len(tr[r]) for r in range(3)) == 0:
            continue
        for _ in range(N_FACT_STEPS):
            fids = [int(rng.choice(tr[0])), int(rng.choice(tr[1])), int(rng.choice(tr[2]))]
            qt = int(rng.integers(3))
            b.train_fact_step([0, 1, 2], fids, roles, fillers, qt, fids[qt])
        nh_ok = nh = 0
        for _ in range(N_EVAL_FACTS):
            fids = rng.choice(F, 3, replace=False)
            bundle = sum(b.bind(roles[r], fillers[fids[r]]) for r in range(3))
            for r in range(3):
                if (r, int(fids[r])) not in train_set:
                    nh_ok += int(native_argmax(b.unbind(bundle, roles[r]), fillers) == fids[r]); nh += 1
        held.append(nh_ok / nh if nh else 0.0)
    return float(np.mean(held)) if held else 0.0


# ============================================================================
# Fixed ±1 self-inverse POSITIVE control (the ~0.989 ceiling on the same codes)
# ============================================================================

def run_fixed_fhrr_control(codes, seed, d_h, A=3):
    fillers = codes[:F]
    D_in = fillers.shape[1]
    roles = make_role_codes(R_MAX, D_in, seed)
    rng = np.random.default_rng(seed * 31 + 5)
    R_proj = rng.standard_normal((D_in, d_h)) / np.sqrt(D_in)
    F_proj = rng.standard_normal((D_in, d_h)) / np.sqrt(D_in)
    role_proj = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    filler_repr = fillers @ F_proj
    fr_unit = filler_repr / (np.linalg.norm(filler_repr, axis=1, keepdims=True) + 1e-12)

    def cleanup(vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return int(np.argmax(fr_unit @ v))

    ok = n = 0
    for _ in range(200):
        fids = rng.choice(F, A, replace=False)
        bundle = sum(role_proj[r] * filler_repr[fids[r]] for r in range(A))
        for r in range(A):
            ok += int(cleanup(bundle * role_proj[r]) == fids[r]); n += 1
    return ok / n if n else 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--attrs", type=str, default="2,3,4")
    ap.add_argument("--d-role", type=str, default="32,64,128")
    ap.add_argument("--n-hidden", type=int, default=1)
    ap.add_argument("--codes", type=str, default="stream", choices=["stream", "neural"])
    ap.add_argument("--run-anticheats", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw",
                                         "_phaseB_fhrr_b_option3_orthogonal_tpr.json"))
    args = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    attrs = [int(a) for a in args.attrs.split(",")]
    droles = [int(d) for d in args.d_role.split(",")]
    t0 = time.time()

    fname = ("_phaseB_stream_codes_320_seed42.npy" if args.codes == "stream"
             else "_phaseB_stream_codes_320_neural_seed42.npy")
    codes_path = os.path.join(_REPO, "research", "findings", "raw", fname)
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True); return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    sub = codes[:F]
    between_cos = float(np.mean([float(sub[i] @ sub[j]) for i in range(F) for j in range(i + 1, F)]))
    chance = 1.0 / F

    print("=" * 110, flush=True)
    print(f"[FHRR-B Option 3 — ORTHOGONAL-TPR + learned cleanup] does a separable (tensor-product) structure make "
          f"the multi-attribute bundle-inverse RECOVERABLE + GENERALIZING without the exact reciprocal?", flush=True)
    print(f"  codes={args.codes} (between-cos {between_cos:.3f}) | F={F} | attrs={attrs} d_role={droles} "
          f"n_hidden={args.n_hidden} | train {N_FACT_STEPS}/split x {N_SPLITS} splits | chance {chance:.3f}",
          flush=True)
    print(f"  refs on this harness: additive {SHALLOW_ADDITIVE} | learned-linear {SHALLOW_LEARNED_LINEAR} | "
          f"dendritic {SHALLOW_DENDRITIC} | Option-1 cleanup {OPTION1_CLEANUP} | fixed-±1 ceiling "
          f"{FIXED_FHRR_CEILING}", flush=True)
    print("=" * 110, flush=True)

    results = {}     # (A, d_role) -> list of per-seed dicts
    for A in attrs:
        for d_role in droles:
            print(f"\n--- A={A} attributes  d_role={d_role} ---", flush=True)
            rows = []
            for s in seeds:
                r = run_seed_cell(codes, s, A=A, d_role=d_role, n_hidden=args.n_hidden)
                rows.append(r)
                print(f"  [seed {s}] A={A} d_role={d_role}: single {r['single_held']:.3f} | BUNDLED train "
                      f"{r['bundle_train']:.3f} | held-out {r['bundle_held']:.3f} | perm-role "
                      f"{r['perm_role_held']:.3f} | moat-gap {r['moat_gap']:+.3f}", flush=True)
            results[(A, d_role)] = rows
            bh = float(np.mean([x["bundle_held"] for x in rows]))
            bt = float(np.mean([x["bundle_train"] for x in rows]))
            sh = float(np.mean([x["single_held"] for x in rows]))
            n_ge = sum(1 for x in rows if x["bundle_held"] >= 0.90)
            print(f"  MEAN A={A} d_role={d_role}: single {sh:.3f} | BUNDLED train {bt:.3f} | held-out {bh:.3f} "
                  f"({n_ge}/{len(rows)} seeds >=0.90)", flush=True)

    # ---- anti-cheat extras ----
    anticheat = {}
    if args.run_anticheats:
        # pick the strongest cell (highest held-out) for the shuffle/lesion controls
        best_cell = max(results.keys(), key=lambda k: float(np.mean([x["bundle_held"] for x in results[k]])))
        bA, bd = best_cell
        print(f"\n--- ANTI-CHEAT: memorization-floor (shuffle train labels) on best cell A={bA} d_role={bd} ---",
              flush=True)
        shuf = [run_seed_cell(codes, s, A=bA, d_role=bd, n_hidden=args.n_hidden, shuffle_train=True) for s in seeds]
        for r, s in zip(shuf, seeds):
            print(f"  [seed {s}] SHUFFLE: BUNDLED held-out {r['bundle_held']:.3f} (must be ~chance {chance:.3f})",
                  flush=True)
        anticheat["shuffle_train_held"] = float(np.mean([r["bundle_held"] for r in shuf]))
        anticheat["shuffle_train_rows"] = shuf

        print(f"\n--- ANTI-CHEAT: LESION (scramble learned cleanup) on best cell A={bA} d_role={bd} ---", flush=True)
        les = [run_seed_cell(codes, s, A=bA, d_role=bd, n_hidden=args.n_hidden, lesion=True) for s in seeds]
        for r, s in zip(les, seeds):
            print(f"  [seed {s}] LESION: BUNDLED held-out {r['bundle_held']:.3f} (must collapse)", flush=True)
        anticheat["lesion_held"] = float(np.mean([r["bundle_held"] for r in les]))
        anticheat["lesion_rows"] = les

        print(f"\n--- LIVE NEGATIVE CONTROL: shallow learned-ADDITIVE bind (ref 0.193) ---", flush=True)
        add = [run_additive_control(codes, s) for s in seeds]
        for r, s in zip(add, seeds):
            print(f"  [seed {s}] ADDITIVE: BUNDLED held-out {r:.3f}", flush=True)
        anticheat["additive_bundled"] = float(np.mean(add)); anticheat["additive_per_seed"] = add

        print(f"\n--- POSITIVE CONTROL: fixed ±1 self-inverse FHRR on the same codes (A=3, D_h=256) ---", flush=True)
        fx = [run_fixed_fhrr_control(codes, s, 256, A=3) for s in seeds]
        for r, s in zip(fx, seeds):
            print(f"  [seed {s}] FIXED-±1: BUNDLED held-out {r:.3f}", flush=True)
        anticheat["fixed_fhrr_bundled"] = float(np.mean(fx)); anticheat["fixed_fhrr_per_seed"] = fx

    # ---- verdict (over the best (A, d_role)) ----
    print(f"\n{'='*110}", flush=True)
    keys = list(results.keys())
    best_key = max(keys, key=lambda k: float(np.mean([x["bundle_held"] for x in results[k]])))
    best_rows = results[best_key]
    best_bh = float(np.mean([x["bundle_held"] for x in best_rows]))
    best_bt = float(np.mean([x["bundle_train"] for x in best_rows]))
    best_sh = float(np.mean([x["single_held"] for x in best_rows]))
    best_pr = float(np.mean([x["perm_role_held"] for x in best_rows]))
    best_moat = float(np.mean([x["moat_gap"] for x in best_rows]))
    n_ge_90 = sum(1 for x in best_rows if x["bundle_held"] >= 0.90)

    print(f"  BEST (A={best_key[0]}, d_role={best_key[1]}): BUNDLED held-out {best_bh:.3f} "
          f"({n_ge_90}/{len(best_rows)} seeds >=0.90), train {best_bt:.3f}, single {best_sh:.3f}", flush=True)
    print(f"  perm-role control {best_pr:.3f} (must be ~chance {chance:.3f}); moat-gap {best_moat:+.3f} (>0)",
          flush=True)
    if args.run_anticheats:
        print(f"  shuffle-train {anticheat['shuffle_train_held']:.3f} (~chance); lesion "
              f"{anticheat['lesion_held']:.3f} (collapse); additive {anticheat['additive_bundled']:.3f} (ref ~0.193); "
              f"fixed-±1 {anticheat['fixed_fhrr_bundled']:.3f}", flush=True)

    # GO requires: generalizing held-out (>=0.90 mean, >=83% seeds), beats the negatives, generalizes (held-out
    # close to train), role-keyed (perm-role ~chance), and the cleanup is load-bearing (lesion collapses).
    add_ref = anticheat.get("additive_bundled", SHALLOW_ADDITIVE)
    les_ref = anticheat.get("lesion_held", 0.0)
    go = (best_bh >= 0.90 and n_ge_90 >= int(np.ceil(0.83 * len(best_rows)))
          and best_bh >= 0.6 * best_bt and best_pr < 0.20 and best_moat > 0.0
          and best_bh > 3.0 * max(add_ref, SHALLOW_DENDRITIC)
          and (not args.run_anticheats or les_ref < 0.40))
    partial = (not go) and best_bh >= 0.40 and best_bh > 1.5 * SHALLOW_DENDRITIC

    if go:
        verdict = "GO"
        print(f"\n  VERDICT: GO — ORTHOGONAL-TPR (a separable, tensor-product structure) + a LEARNED cleanup "
              f"RECOVERS + GENERALIZES the multi-attribute bundle-inverse WITHOUT the exact reciprocal: held-out "
              f"{best_bh:.3f} >> the shallow learned NEGATIVEs (additive {add_ref:.3f}/dendritic {SHALLOW_DENDRITIC}), "
              f"generalizing (held-out {best_bh:.3f} vs train {best_bt:.3f}), role-keyed (perm-role {best_pr:.3f} ~ "
              f"chance). ==> FHRR-B's multi-attribute capability closes via LEARNED codes + a FIXED separable "
              f"(orthogonal-role) structure + a LEARNED read-out. HONEST: the role tags are a FIXED structural "
              f"choice (not a learned bind); a DIFFERENT fixed structure makes multi-attribute recovery learnable + "
              f"generalizing. Hand the controller the spiking-confirm route.", flush=True)
    elif partial:
        verdict = "PARTIAL"
        print(f"\n  VERDICT: PARTIAL — orthogonal-TPR LIFTS the bundle-inverse above the shallow NEGATIVEs (held-out "
              f"{best_bh:.3f} vs dendritic {SHALLOW_DENDRITIC}) but short of robust generalization. A real lever, not "
              f"parity. NOT a closed boundary (owner's rule): informs the converged structural-primitive synthesis.",
              flush=True)
    else:
        verdict = "NEGATIVE"
        print(f"\n  VERDICT: NEGATIVE — orthogonal-TPR + a learned cleanup does NOT generalize the multi-attribute "
              f"bundle-inverse (best held-out {best_bh:.3f}). Even with a separable, no-crosstalk tensor-product "
              f"structure (which REMOVES the exact-reciprocal requirement entirely), the LEARNED read-out does not "
              f"systematically recombine roles with held-out fillers. NOT a closed boundary (owner's rule): with "
              f"Option-1 cleanup GO + Option-2 deep-bind NEGATIVE + Option-3 orthogonal-TPR NEGATIVE, the evidence "
              f"CONVERGES on a genuine structural finding for the controller — the role-filler BIND is a FIXED "
              f"STRUCTURAL neural primitive (binding-by-coincidence / dendritic multiplication), NOT a learnable host "
              f"op. Closing FHRR-B = learned codes (done) + learned cleanup (Option-1 GO) + a fixed structural bind.",
              flush=True)

    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*110}", flush=True)

    out = {
        "meta": {"codes": args.codes, "between_cos": between_cos, "F": F, "attrs": attrs, "d_role": droles,
                 "n_hidden": args.n_hidden, "seeds": seeds, "n_fact_steps": N_FACT_STEPS,
                 "n_splits": N_SPLITS, "lr": LR},
        "refs": {"additive": SHALLOW_ADDITIVE, "learned_linear": SHALLOW_LEARNED_LINEAR,
                 "dendritic": SHALLOW_DENDRITIC, "option1_cleanup": OPTION1_CLEANUP,
                 "fixed_fhrr_ceiling": FIXED_FHRR_CEILING, "chance": chance},
        "by_cell": {f"A{k[0]}_dr{k[1]}": results[k] for k in keys},
        "means": {f"A{k[0]}_dr{k[1]}": {
            "single_held": float(np.mean([x["single_held"] for x in results[k]])),
            "bundle_train": float(np.mean([x["bundle_train"] for x in results[k]])),
            "bundle_held": float(np.mean([x["bundle_held"] for x in results[k]])),
            "perm_role_held": float(np.mean([x["perm_role_held"] for x in results[k]])),
            "moat_gap": float(np.mean([x["moat_gap"] for x in results[k]])),
            "n_seeds_ge_0.90": sum(1 for x in results[k] if x["bundle_held"] >= 0.90),
        } for k in keys},
        "best": {"A": best_key[0], "d_role": best_key[1], "bundle_held": best_bh}, "verdict": verdict,
        "anticheat": anticheat,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
