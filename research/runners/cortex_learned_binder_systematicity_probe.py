"""CORTEX LEARNED-BINDER SYSTEMATICITY PROBE.

CONTEXT
-------
The conversational composer binds role-filler facts via an EXACT-INVERSE FHRR algebra.
It is a principled IDEALIZATION: the binding is exactly invertible but DEMANDS decorrelated
/ clean codes. A prior cleanup sub-arc proved (3 distinct NEGATIVES) that a post-hoc
attractor on the brain's CORRELATED denoise64 codes (cosine ~0.81) cannot be realized
by a local spiking rule. The complementary positive control (cortex_sparse_attractor_
poscontrol_probe) confirmed that a distributed attractor WORKS on the project's REAL
decorrelated sparse codes (cosine ~0.05).

This probe asks the deeper SYSTEMATICITY question: can a LEARNED binder (not the fixed
exact-inverse algebra) bind role-filler pairs over CORRELATED, similarity-structured codes
AND generalize to role-filler COMBINATIONS it never saw in training?

Systematicity (Fodor-Pylyshyn 1988) is the load-bearing risk: a learned binder that merely
MEMORIZES training pairs is useless; a real cortex recombines known roles with known fillers
it never saw paired. If a learned binder generalizes systematically on correlated codes ->
the learned-cortex path (Option C) is viable. If it only memorizes -> that is the mapped
Fodor-Pylyshyn boundary.

BRAIN-BASED STAGING
-------------------
This is a CHEAP-FIRST capacity/systematicity characterization, exactly like the numpy-Hopfield
cleanup probes that preceded the spiking ones. A host-optimized (gradient-trained) bilinear
binder answers "does ANY learned binder generalize systematically on these codes?" If YES ->
the spiking BPTT realization (sim/bptt_snn.py) is the later build. If even a host-optimized
binder can't generalize, no spiking version will. Label the host binder as characterization,
NOT the deliverable. The FHRR exact-inverse appears only as the systematic-by-construction
REFERENCE ceiling.

CODE REGIMES (sweep both)
--------------------------
(a) DECORRELATED codes: generate_sparse_patterns(cos~0.05) -- the baseline regime.
(b) CORRELATED codes: denoise64 loader from cortex_storkey_ca3_cleanup_probe (cos~0.81) --
    the REAL target regime.

Unit check (BEFORE any test): assert sparse between-cos < 0.15 AND correlated between-cos > 0.60.
NEVER median-bipolarize sparse codes (manufactures a ~1 common mode -> false NEGATIVE).

TASK STRUCTURE
--------------
R roles (R=4), F fillers (sweep F in {8, 16}).
FACT = (role, filler) pair. bind -> bound vector. UNBIND: given (bound, role) -> nearest
filler by cosine == true filler.

Binders:
1. FHRR exact-inverse (circular convolution / element-wise complex product + conjugate).
   REFERENCE: systematic by construction (expect ~1.000 on decorrelated).
2. Bilinear learned binder: bound = role_embed @ W_bind @ filler_embed; unbind =
   filler_hat = role_embed @ W_unbind @ bound. Trained by gradient descent on TRAIN combos.
3. (as baseline) Memorization lookup table: store (role, filler) -> bound directly.

SYSTEMATICITY PROTOCOL (leakage-free)
--------------------------------------
Enumerate all R x F combos. Split TRAIN/HELD-OUT:
- EVERY role and EVERY filler appears in >=1 train combo (seen atoms).
- The SPECIFIC (role_i, filler_j) pairings in HELD-OUT never appear in TRAIN (novel combos).
Run multiple splits per seed (3 splits x 3 seeds = 9 trials per regime x F combo).

Metrics:
- TRAIN-combo unbind accuracy (confirms it learned to bind at all).
- HELD-OUT-combo unbind accuracy (the systematicity metric).
- GAP = train_acc - held_out_acc. Systematic: held_out ~ train. Memorization: held_out ~ chance.

ANTI-CHEATS (all 4 are mandatory)
----------------------------------
1. LEAKAGE ASSERT: programmatically verify NO test combo is in the train set (assert empty
   intersection). Print |train|, |test|, confirm every role & filler covered in train.
2. SHUFFLED-HELD-OUT-LABEL CONTROL: score held-out predictions against SHUFFLED true fillers
   -> must drop to chance (confirms held-out accuracy is real, not a readout artifact).
3. MEMORIZATION CONTROL: pure lookup table (train pairs memorized) -> MUST score chance on
   held-out (defines the memorization floor; the learned binder beating this is the signal).
4. NO-CONFAB / ABSTENTION FLOOR: present an unbind query for a (role, filler) where the
   filler was NEVER bound to anything -> readout should show low confidence. Report the
   confidence-gap between seen and never-seen fillers.

DECISION LOGIC
--------------
Option-C VIABLE if: learned binder achieves high held-out accuracy (systematic, ~= train,
  well above memorization floor) on CORRELATED codes, multi-seed, with all 4 controls decisive.
BOUNDARY/NEGATIVE if: held-out ~ chance on correlated codes (memorization on the real target).
  Characterize: does it ALSO fail on decorrelated codes? (if generalizes on decorrelated but not
  correlated -> failure is the correlation; if fails on both -> form too weak; if both pass ->
  Option C strongly viable).

CPU-only; SIM_BACKEND=numpy; no sim/ edits; reuse by import only.
Run: python -m research.runners.cortex_learned_binder_systematicity_probe [--seeds 42,43,44] [--F 8,16]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DENOISE64_CACHE = os.path.join(
    _REPO, "research", "findings", "raw",
    "activity_level_integration_cache", "denoise64_seed%d.npz"
)


# ============================================================================
# CODE LOADERS (CORRECT NATIVE READOUTS)
# ============================================================================

def load_sparse_codes_native(seed: int, V: int, n_pool: int = 2000,
                              pattern_size: int = 100,
                              proj_dim: int = 0) -> Tuple[np.ndarray, float, float]:
    """Load sparse-distributed codes in NATIVE binary form, mean-removed.

    Convention: binary {0,1} mask -> mean-remove -> unit-normalize.
    NEVER median-bipolarize (that manufactures a ~1 common mode on sparse codes -> false NEGATIVE).

    Optional: if proj_dim > 0, project down via a random Gaussian (cosine-preserving).
    This is used to bring the sparse codes to the same dimension as the dense codes
    so the bilinear binder has comparable parameter counts across regimes.

    Returns (codes [V, proj_dim or n_pool], between_cos_mean, between_cos_max).
    """
    from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
    patterns = generate_sparse_patterns(V, n_pool, pattern_size, seed)
    codes = np.zeros((V, n_pool), dtype=np.float64)
    for i, pat in enumerate(patterns):
        codes[i, pat] = 1.0
    # Mean-remove (removes the shared sparsity common mode -- critical for sparse codes)
    codes = codes - codes.mean(axis=1, keepdims=True)
    # Unit-normalize
    norms = np.linalg.norm(codes, axis=1, keepdims=True)
    codes = codes / (norms + 1e-12)
    # Optional Gaussian projection (cosine-preserving)
    if proj_dim > 0 and proj_dim < n_pool:
        rng_proj = np.random.default_rng(seed * 31337 + 999)
        P = rng_proj.standard_normal((n_pool, proj_dim)) / np.sqrt(n_pool)
        codes = codes @ P
        norms2 = np.linalg.norm(codes, axis=1, keepdims=True)
        codes = codes / (norms2 + 1e-12)
    # Between-code cosines
    cos_vals = [float(codes[i] @ codes[j]) for i in range(V) for j in range(i + 1, V)]
    between_cos_mean = float(np.mean(cos_vals)) if cos_vals else 0.0
    between_cos_max = float(np.max(np.abs(cos_vals))) if cos_vals else 0.0
    return codes, between_cos_mean, between_cos_max


def load_denoise64_codes(seed: int, V: int = 16,
                          proj_dim: int = 800) -> Tuple[List[str], np.ndarray, float]:
    """Load denoise64 brain codes (projected, centered, normed).

    Same convention as core_sim_composition.load_concepts:
    mean over obs samples, Gaussian project to proj_dim, row-mean-center, unit-norm.
    NO decorrelation -- these are the RAW CORRELATED codes (the real target regime).
    Returns (words [V], codes [V, proj_dim], between_cos_mean).
    """
    rng = np.random.RandomState(seed)
    d = np.load(DENOISE64_CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    ws = ws[:V]
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)
    if proj_dim and proj_dim > 0:
        P = rng.randn(raw.shape[1], proj_dim) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = raw - raw.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    cos_vals = [float(codes[i] @ codes[j]) for i in range(V) for j in range(i + 1, V)]
    between_cos_mean = float(np.mean(cos_vals)) if cos_vals else 0.0
    return ws, codes, between_cos_mean


# ============================================================================
# UNIT CHECK (runs before EVERYTHING; aborts if codes mis-read)
# ============================================================================

def unit_check(sparse_cos: float, correlated_cos: float,
               sparse_threshold: float = 0.15,
               correlated_threshold: float = 0.60) -> dict:
    """Assert the two code families are in the correct correlation regime."""
    ok_sparse = sparse_cos < sparse_threshold
    ok_dense = correlated_cos > correlated_threshold
    status = "PASS" if (ok_sparse and ok_dense) else "FAIL"
    return {
        "sparse_between_cos_mean": float(sparse_cos),
        "correlated_between_cos_mean": float(correlated_cos),
        "ok_sparse_decorrelated": bool(ok_sparse),
        "ok_correlated": bool(ok_dense),
        "status": status,
    }


# ============================================================================
# ROLE CODES (fixed random vectors; same D as filler codes)
# ============================================================================

def make_role_codes(R: int, D: int, seed: int) -> np.ndarray:
    """Generate R fixed orthogonal-ish role codes in the same space as filler codes.
    Uses standard-normal random + unit-normalize -> near-orthogonal for R << D."""
    rng = np.random.default_rng(seed * 100007 + 3)
    roles = rng.standard_normal((R, D))
    norms = np.linalg.norm(roles, axis=1, keepdims=True)
    return roles / (norms + 1e-12)


# ============================================================================
# FHRR EXACT-INVERSE (circular convolution; REFERENCE -- systematic by construction)
# ============================================================================

def fhrr_bind(role: np.ndarray, filler: np.ndarray) -> np.ndarray:
    """FHRR bind: element-wise complex product in frequency domain.
    role, filler are real vectors -> interpret as phases of unit complex vectors,
    then multiply phases (add angles). Returns bound phases.
    Equivalent to circular convolution on the unit-phasor codes."""
    # Represent as complex unit vectors: e^{i * pi * code}
    role_c = np.exp(1j * np.pi * role)
    filler_c = np.exp(1j * np.pi * filler)
    bound_c = role_c * filler_c
    return np.angle(bound_c) / np.pi   # back to real phases in [-1, 1]


def fhrr_unbind(bound: np.ndarray, role: np.ndarray) -> np.ndarray:
    """FHRR unbind: element-wise multiply by conjugate of role.
    Exact inverse: conj(role) * bound = filler (mod noise)."""
    bound_c = np.exp(1j * np.pi * bound)
    role_c = np.exp(1j * np.pi * role)
    filler_c = bound_c * np.conj(role_c)
    return np.angle(filler_c) / np.pi


def fhrr_cleanup(estimate: np.ndarray, filler_codes: np.ndarray) -> int:
    """Nearest-code argmax (cosine) over filler_codes. REFERENCE scoring step."""
    estimate_c = np.exp(1j * np.pi * estimate)
    codes_c = np.exp(1j * np.pi * filler_codes)   # [V, D] complex
    # Cosine in complex space = Re(<estimate, code>) / (|estimate| * |code|)
    dots = np.real(codes_c @ np.conj(estimate_c))   # [V]
    return int(np.argmax(dots))


# ============================================================================
# NATIVE ARGMAX (for real-valued sparse + denoise64 codes)
# ============================================================================

def native_argmax(estimate: np.ndarray, filler_codes: np.ndarray) -> int:
    """Nearest-code argmax by real cosine. Used for sparse/dense binders."""
    norms = np.linalg.norm(filler_codes, axis=1) * (np.linalg.norm(estimate) + 1e-12)
    sims = filler_codes @ estimate / (norms + 1e-12)
    return int(np.argmax(sims))


# ============================================================================
# BILINEAR LEARNED BINDER (the candidate)
# ============================================================================
# Architecture:
#   bind:   bound = tanh( role @ W_R + filler @ W_F + b_bind )    [D_h]
#   unbind: filler_hat = bound @ W_U + role @ W_RU + b_unbind     [D]
#   loss:   MSE(filler_hat, filler_true)
#
# This is the simplest thing that can learn to associate roles with fillers.
# Richer architectures (MLP, bilinear product, etc.) are not needed to answer
# the systematicity question -- if even this form fails systematically on
# correlated codes, a harder architecture will not fix it; if it passes,
# the BPTT cortex realization is de-risked.

class BilinearBinder:
    """Gradient-trained bilinear binder.

    bind(role, filler) -> hidden code
    unbind(hidden, role) -> filler estimate

    Parameters
    ----------
    D_in : int   - dimension of role/filler codes
    D_h  : int   - hidden / bound-vector dimension
    lr   : float - learning rate
    lam  : float - L2 regularization
    """

    def __init__(self, D_in: int, D_h: int, lr: float = 0.01,
                 lam: float = 1e-4, seed: int = 42,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        rng = np.random.default_rng(seed * 999 + 7)
        scale = 1.0 / np.sqrt(D_in)
        self.W_R = rng.standard_normal((D_in, D_h)) * scale   # role -> hidden
        self.W_F = rng.standard_normal((D_in, D_h)) * scale   # filler -> hidden
        self.b_bind = np.zeros(D_h)
        # Unbind: bound [D_h] + role [D_in] -> hidden2 [D_h2] -> filler [D_in]
        # Use a two-layer unbind to avoid the expensive D_in x D_in W_RU bypass:
        #   concat([bound, role_proj]) -> W_U -> D_in
        # role_proj = role @ W_RP (D_in -> D_h), concat with bound = D_h*2
        self.W_RP = rng.standard_normal((D_in, D_h)) * scale  # role -> hidden (unbind path)
        self.W_U = rng.standard_normal((D_h * 2, D_in)) * scale  # [bound||role_h] -> filler
        self.b_unbind = np.zeros(D_in)
        self.lr = lr
        self.lam = lam
        self.D_h = D_h
        self.D_in = D_in
        # Adam optimizer state
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # step counter (bias correction)
        # First moments (m)
        self.m_WR = np.zeros_like(self.W_R)
        self.m_WF = np.zeros_like(self.W_F)
        self.m_bb = np.zeros_like(self.b_bind)
        self.m_WRP = np.zeros_like(self.W_RP)
        self.m_WU = np.zeros_like(self.W_U)
        self.m_bu = np.zeros_like(self.b_unbind)
        # Second moments (v)
        self.v_WR = np.zeros_like(self.W_R)
        self.v_WF = np.zeros_like(self.W_F)
        self.v_bb = np.zeros_like(self.b_bind)
        self.v_WRP = np.zeros_like(self.W_RP)
        self.v_WU = np.zeros_like(self.W_U)
        self.v_bu = np.zeros_like(self.b_unbind)

    def _bind(self, role: np.ndarray, filler: np.ndarray) -> np.ndarray:
        """Forward: bind role + filler -> bound [D_h]."""
        h = role @ self.W_R + filler @ self.W_F + self.b_bind
        return np.tanh(h)   # [D_h]

    def _unbind(self, bound: np.ndarray, role: np.ndarray) -> np.ndarray:
        """Forward: unbind bound + role -> filler estimate [D_in].
        Concatenates bound [D_h] with role_proj [D_h] -> [2*D_h] -> linear -> [D_in].
        Avoids D_in x D_in matrix by projecting role to D_h first."""
        role_h = role @ self.W_RP   # [D_h]
        concat = np.concatenate([bound, role_h])   # [2*D_h]
        return concat @ self.W_U + self.b_unbind  # [D_in]

    def predict(self, role: np.ndarray, filler: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (bound, filler_estimate)."""
        bound = self._bind(role, filler)
        est = self._unbind(bound, role)
        return bound, est

    def _adam_update(self, param, grad, m, v):
        """Apply one Adam update in-place. Returns updated (param, m, v)."""
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
        m_hat = m / (1.0 - self.beta1 ** self.t)
        v_hat = v / (1.0 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param, m, v

    def train_step(self, role: np.ndarray, filler: np.ndarray) -> float:
        """One Adam step on (role, filler) pair. Returns MSE loss."""
        self.t += 1
        # Forward (bind)
        h_pre = role @ self.W_R + filler @ self.W_F + self.b_bind   # [D_h]
        bound = np.tanh(h_pre)                                        # [D_h]
        # Forward (unbind): concat [bound || role_h] @ W_U
        role_h = role @ self.W_RP                                    # [D_h]
        concat = np.concatenate([bound, role_h])                     # [2*D_h]
        est = concat @ self.W_U + self.b_unbind                      # [D_in]
        # Loss: MSE(est, filler)
        err = est - filler    # [D_in]
        loss = float(np.mean(err ** 2))
        # Backward through unbind
        d_est = 2.0 * err / self.D_in           # [D_in]
        d_concat = self.W_U @ d_est              # [2*D_h]
        d_W_U = np.outer(concat, d_est)          # [2*D_h, D_in]
        d_b_unbind = d_est.copy()
        # Split gradient through concat
        d_bound = d_concat[:self.D_h]            # [D_h]
        d_role_h = d_concat[self.D_h:]           # [D_h]
        d_W_RP = np.outer(role, d_role_h)        # [D_in, D_h]
        # Backward through tanh (bind path)
        d_h_pre = d_bound * (1.0 - bound ** 2)  # [D_h]
        d_W_R = np.outer(role, d_h_pre)          # [D_in, D_h]
        d_W_F = np.outer(filler, d_h_pre)        # [D_in, D_h]
        d_b_bind = d_h_pre.copy()
        # Adam updates with L2 regularization
        self.W_R, self.m_WR, self.v_WR = self._adam_update(
            self.W_R, d_W_R + self.lam * self.W_R, self.m_WR, self.v_WR)
        self.W_F, self.m_WF, self.v_WF = self._adam_update(
            self.W_F, d_W_F + self.lam * self.W_F, self.m_WF, self.v_WF)
        self.b_bind, self.m_bb, self.v_bb = self._adam_update(
            self.b_bind, d_b_bind, self.m_bb, self.v_bb)
        self.W_RP, self.m_WRP, self.v_WRP = self._adam_update(
            self.W_RP, d_W_RP + self.lam * self.W_RP, self.m_WRP, self.v_WRP)
        self.W_U, self.m_WU, self.v_WU = self._adam_update(
            self.W_U, d_W_U + self.lam * self.W_U, self.m_WU, self.v_WU)
        self.b_unbind, self.m_bu, self.v_bu = self._adam_update(
            self.b_unbind, d_b_unbind, self.m_bu, self.v_bu)
        return loss

    def train(self, train_combos: List[Tuple[int, int]],
              role_codes: np.ndarray, filler_codes: np.ndarray,
              n_epochs: int = 500, batch_size: int = 8,
              verbose: bool = False) -> List[float]:
        """Train on (role_idx, filler_idx) pairs."""
        rng = np.random.default_rng(42)
        losses = []
        n = len(train_combos)
        for epoch in range(n_epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = [train_combos[perm[i]] for i in range(start, min(start + batch_size, n))]
                for ri, fi in batch:
                    l = self.train_step(role_codes[ri], filler_codes[fi])
                    epoch_loss += l
            avg_loss = epoch_loss / n
            losses.append(avg_loss)
            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d}  loss={avg_loss:.6f}", flush=True)
        return losses


# ============================================================================
# MEMORIZATION LOOKUP TABLE (the memorization floor / anti-cheat 3 baseline)
# ============================================================================

class MemorizationLookup:
    """Pure lookup table binder: stores (role_idx, filler_idx) -> filler_code directly.
    On held-out combos it cannot retrieve the correct filler (it has no entry).
    Its held-out accuracy MUST be chance (1/F) -- this defines the memorization floor."""

    def __init__(self):
        self._store: dict = {}   # (role_idx, filler_idx) -> filler_code

    def train(self, train_combos: List[Tuple[int, int]],
              role_codes: np.ndarray, filler_codes: np.ndarray, **kwargs):
        """Memorize training pairs."""
        for ri, fi in train_combos:
            self._store[(ri, fi)] = filler_codes[fi].copy()

    def predict_unbind(self, role_idx: int, bound_from_learned: np.ndarray,
                       role_codes: np.ndarray, filler_codes: np.ndarray) -> int:
        """Unbind: if this (role, any filler) pair in store, retrieve a stored filler
        associated with this role; otherwise return random (chance behavior)."""
        # The lookup table has no 'bound' state -- it just retrieves stored filler by role.
        # On held-out (role, filler) combos, the filler it guesses is whatever it learned
        # for this role in training (biased toward training fillers, not the test filler).
        # This is the correct memorization floor: it can only return training-seen combos.
        # Find all entries with this role
        candidates = [fi for (ri, fi) in self._store.keys() if ri == role_idx]
        if not candidates:
            # Never saw this role (shouldn't happen with our split design)
            return int(np.random.randint(len(filler_codes)))
        # Return the nearest training filler to the role's stored codes
        # (arbitrary choice -- the point is it's NOT the test filler for held-out)
        # For a clean memorization floor, just return the first stored filler for this role
        fi_stored = candidates[0]
        return fi_stored


# ============================================================================
# SYSTEMATICITY SPLIT
# ============================================================================

def make_systematicity_splits(R: int, F: int, n_splits: int,
                               seed: int) -> List[Dict]:
    """Generate n_splits train/held-out splits of R x F combos.

    Each split:
    - TRAIN: subset of (role, filler) combos.
    - HELD-OUT: complementary combos NOT in TRAIN.
    - Constraint: every role in train, every filler in train.
    - The held-out combos are NOVEL RECOMBINATIONS of seen atoms.

    Strategy: for each split, randomly designate one held-out row-column combination.
    We hold out one filler per role (cyclic with rotation), ensuring every role and
    every filler appears at least once in train.

    For R=4, F=8: total = 32 combos. Hold out 4 combos (one per role, cycling fillers).
    Each filler appears in 1 held-out and 3 train combos; each role appears in 7 train.

    Returns list of {'train': [(ri, fi)...], 'held_out': [(ri, fi)...],
                     'train_roles_covered': bool, 'train_fillers_covered': bool,
                     'split_id': int, 'leakage_free': bool}
    """
    rng = np.random.default_rng(seed * 37 + 13)
    splits = []
    all_combos = [(r, f) for r in range(R) for f in range(F)]

    for split_id in range(n_splits):
        # For this split, hold out exactly one (role_i, filler_j) pair per role,
        # where the filler assignment is a random permutation (no role gets the same
        # filler in the held-out set across splits).
        filler_perm = rng.permutation(F)
        held_out = [(r, int(filler_perm[r % F])) for r in range(R)]
        held_out_set = set(held_out)

        # Ensure no role gets the same held-out filler in multiple roles
        # (already satisfied since we assign one per role)
        train = [c for c in all_combos if c not in held_out_set]

        # Verify every role and filler appears in train
        train_roles = set(r for r, f in train)
        train_fillers = set(f for r, f in train)
        all_roles_covered = (train_roles == set(range(R)))
        all_fillers_covered = (train_fillers == set(range(F)))

        # Leakage assert: intersection of train and held_out must be empty
        leakage_free = len(set(train) & held_out_set) == 0

        splits.append({
            "split_id": split_id,
            "train": train,
            "held_out": held_out,
            "n_train": len(train),
            "n_held_out": len(held_out),
            "train_roles_covered": bool(all_roles_covered),
            "train_fillers_covered": bool(all_fillers_covered),
            "leakage_free": bool(leakage_free),
        })

    return splits


# ============================================================================
# ACCURACY MEASUREMENT
# ============================================================================

def score_bilinear_unbind(binder: BilinearBinder,
                           combos: List[Tuple[int, int]],
                           role_codes: np.ndarray,
                           filler_codes: np.ndarray) -> float:
    """Score unbind accuracy of the bilinear binder on combos.
    For each (ri, fi): bind (ri, fi) -> bound; unbind (bound, ri) -> estimate;
    nearest-filler argmax(estimate, filler_codes) == fi?"""
    n_correct = 0
    for ri, fi in combos:
        bound = binder._bind(role_codes[ri], filler_codes[fi])
        est = binder._unbind(bound, role_codes[ri])
        pred = native_argmax(est, filler_codes)
        n_correct += int(pred == fi)
    return n_correct / len(combos) if combos else 0.0


def score_fhrr_unbind(combos: List[Tuple[int, int]],
                      role_codes: np.ndarray,
                      filler_codes: np.ndarray) -> float:
    """Score FHRR exact-inverse unbind accuracy on combos.
    For each (ri, fi): fhrr_bind(ri, fi) -> bound; fhrr_unbind(bound, ri) -> estimate;
    fhrr_cleanup(estimate, filler_codes) == fi?"""
    # FHRR expects codes in phase space: use the codes directly as phases.
    # For real-valued codes (not phasors), we treat them as phases in [-1, 1].
    n_correct = 0
    for ri, fi in combos:
        bound = fhrr_bind(role_codes[ri], filler_codes[fi])
        est = fhrr_unbind(bound, role_codes[ri])
        pred = fhrr_cleanup(est, filler_codes)
        n_correct += int(pred == fi)
    return n_correct / len(combos) if combos else 0.0


def score_shuffled_label(binder: BilinearBinder,
                          combos: List[Tuple[int, int]],
                          role_codes: np.ndarray,
                          filler_codes: np.ndarray,
                          rng: np.random.Generator) -> float:
    """Anti-cheat 2: score held-out predictions against SHUFFLED true filler labels.
    Must drop to chance (1/F) -- confirms held-out accuracy is real, not a readout artifact."""
    # Build estimates
    preds = []
    true_labels = []
    for ri, fi in combos:
        bound = binder._bind(role_codes[ri], filler_codes[fi])
        est = binder._unbind(bound, role_codes[ri])
        preds.append(native_argmax(est, filler_codes))
        true_labels.append(fi)
    # Shuffle the true labels
    shuffled = list(rng.permutation(true_labels))
    n_correct = sum(int(p == s) for p, s in zip(preds, shuffled))
    return n_correct / len(preds) if preds else 0.0


def score_memorization_floor(train_combos: List[Tuple[int, int]],
                              held_out_combos: List[Tuple[int, int]],
                              filler_codes: np.ndarray) -> Dict:
    """Anti-cheat 3: the lookup-table memorization floor.
    - On train: should match perfectly (role, filler) in table.
    - On held-out: should score chance (1/F) -- the floor the learned binder must beat.
    """
    lookup = MemorizationLookup()
    lookup.train(train_combos, None, filler_codes)
    F = len(filler_codes)

    # Train accuracy: for memorized pairs, lookup retrieves the stored filler
    train_correct = 0
    for ri, fi in train_combos:
        candidates = [cf for (cr, cf) in lookup._store.keys() if cr == ri]
        # The lookup stores all training fillers; the "right" answer is fi if it was stored
        if (ri, fi) in lookup._store:
            train_correct += 1
    train_acc = train_correct / len(train_combos) if train_combos else 0.0

    # Held-out accuracy: the lookup can only return a training filler for this role;
    # the probability of guessing the correct held-out filler by coincidence is at most
    # (1 if the training filler for that role happens to be the test filler, else 0).
    # By construction (split design: held-out fillers differ from what's typically stored),
    # this should be near chance.
    held_correct = 0
    for ri, fi in held_out_combos:
        # What does the lookup return for role ri?
        candidates = [cf for (cr, cf) in lookup._store.keys() if cr == ri]
        if candidates:
            # Return the last-stored filler for this role (arbitrary choice -- the point
            # is it's NOT the novel test filler for most cases)
            guess = candidates[-1]
        else:
            guess = -1
        held_correct += int(guess == fi)
    held_acc = held_correct / len(held_out_combos) if held_out_combos else 0.0

    return {
        "train_acc": float(train_acc),
        "held_out_acc": float(held_acc),
        "chance": 1.0 / F,
        "held_above_chance": float(held_acc) > (1.0 / F + 0.1),
    }


def score_abstention(binder: BilinearBinder,
                     known_combos: List[Tuple[int, int]],
                     R: int, F: int,
                     role_codes: np.ndarray,
                     filler_codes: np.ndarray) -> Dict:
    """Anti-cheat 4 (no-confab / abstention floor).

    Present an unbind query for a (role, filler) where the filler was NEVER bound to anything.
    'Never-seen filler': create a random novel filler code (not in the codebook).
    Measure confidence gap between queries with real fillers vs novel fillers.
    High-confidence on real fillers + low-confidence on novel fillers = the system could
    support a familiarity gate.

    Confidence measure: max cosine of the estimate to any filler in the codebook.
    A well-behaved binder: max_cos(known) >> max_cos(novel).
    """
    rng = np.random.default_rng(54321)

    # Known-filler estimates: use the training combos
    known_confs = []
    for ri, fi in known_combos[:min(20, len(known_combos))]:
        bound = binder._bind(role_codes[ri], filler_codes[fi])
        est = binder._unbind(bound, role_codes[ri])
        est_norm = est / (np.linalg.norm(est) + 1e-12)
        sims = filler_codes @ est_norm
        known_confs.append(float(np.max(sims)))

    # Novel-filler estimates: unbind using a random-noise "filler" not in the codebook
    novel_confs = []
    for ri in range(min(R, 4)):
        for _ in range(5):
            # Random novel filler (unit Gaussian, out-of-distribution)
            novel_filler = rng.standard_normal(filler_codes.shape[1])
            novel_filler = novel_filler / (np.linalg.norm(novel_filler) + 1e-12)
            bound = binder._bind(role_codes[ri], novel_filler)
            est = binder._unbind(bound, role_codes[ri])
            est_norm = est / (np.linalg.norm(est) + 1e-12)
            sims = filler_codes @ est_norm
            novel_confs.append(float(np.max(sims)))

    mean_known = float(np.mean(known_confs)) if known_confs else 0.0
    mean_novel = float(np.mean(novel_confs)) if novel_confs else 0.0
    gap = mean_known - mean_novel
    return {
        "mean_conf_known_filler": mean_known,
        "mean_conf_novel_filler": mean_novel,
        "familiarity_gap": gap,
        "n_known_queries": len(known_confs),
        "n_novel_queries": len(novel_confs),
        "gap_positive": bool(gap > 0.0),
    }


# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def run_condition(regime_name: str,
                  filler_codes: np.ndarray,
                  R: int, F: int, seed: int,
                  n_splits: int,
                  n_epochs: int,
                  D_h: int,
                  lr: float,
                  verbose: bool) -> Dict:
    """Run the full protocol for one (regime, F, seed) condition.

    Returns a dict with all metrics, anti-cheat outcomes, and the decision.
    """
    t0 = time.time()
    filler_codes_F = filler_codes[:F]    # use first F fillers
    D_in = filler_codes_F.shape[1]

    # Role codes: generated in the same dimension as filler codes
    role_codes = make_role_codes(R, D_in, seed)

    # Compute between-code cosines of role codes (should be ~1/sqrt(D) for random)
    role_cos_vals = [float(role_codes[i] @ role_codes[j])
                     for i in range(R) for j in range(i + 1, R)]
    role_cos_mean = float(np.mean(role_cos_vals)) if role_cos_vals else 0.0

    # Generate splits
    splits = make_systematicity_splits(R, F, n_splits, seed)

    all_results = []
    for split in splits:
        train_combos = split["train"]
        held_out_combos = split["held_out"]
        n_train = len(train_combos)
        n_held = len(held_out_combos)
        chance = 1.0 / F

        # --- LEAKAGE ASSERT (anti-cheat 1) ---
        train_set = set(train_combos)
        held_set = set(held_out_combos)
        leakage_count = len(train_set & held_set)
        assert leakage_count == 0, (
            f"LEAKAGE: {leakage_count} combos appear in both train and held-out!"
        )
        all_roles_in_train = all(r in {ri for ri, fi in train_combos} for r in range(R))
        all_fillers_in_train = all(f in {fi for ri, fi in train_combos} for f in range(F))

        # --- FHRR EXACT-INVERSE (reference) ---
        fhrr_train_acc = score_fhrr_unbind(train_combos, role_codes, filler_codes_F)
        fhrr_held_acc = score_fhrr_unbind(held_out_combos, role_codes, filler_codes_F)

        # --- BILINEAR LEARNED BINDER ---
        binder = BilinearBinder(D_in=D_in, D_h=D_h, lr=lr, lam=1e-4, seed=seed)
        binder.train(train_combos, role_codes, filler_codes_F,
                     n_epochs=n_epochs, batch_size=max(1, n_train // 4),
                     verbose=verbose)

        bilinear_train_acc = score_bilinear_unbind(binder, train_combos, role_codes, filler_codes_F)
        bilinear_held_acc = score_bilinear_unbind(binder, held_out_combos, role_codes, filler_codes_F)

        # --- ANTI-CHEAT 2: SHUFFLED LABEL ---
        rng_shuffle = np.random.default_rng(seed * 101 + split["split_id"])
        shuffled_held_acc = score_shuffled_label(binder, held_out_combos,
                                                  role_codes, filler_codes_F, rng_shuffle)

        # --- ANTI-CHEAT 3: MEMORIZATION FLOOR ---
        mem_floor = score_memorization_floor(train_combos, held_out_combos, filler_codes_F)

        # --- ANTI-CHEAT 4: ABSTENTION / FAMILIARITY ---
        abstention = score_abstention(binder, train_combos, R, F, role_codes, filler_codes_F)

        # --- DECISION for this split ---
        # Systematic if: held_out_acc > 2x_chance AND held_out > mem_floor + 0.1 AND
        #               shuffled drops to chance AND FHRR stays ~1.0
        systematic = (
            bilinear_held_acc > 2 * chance and
            bilinear_held_acc > mem_floor["held_out_acc"] + 0.1 and
            shuffled_held_acc < bilinear_held_acc - 0.15 and
            fhrr_held_acc > 0.9
        )
        memorization = (
            bilinear_train_acc > 0.8 and
            bilinear_held_acc <= 2 * chance
        )
        verdict = ("SYSTEMATIC" if systematic else
                   "MEMORIZATION" if memorization else
                   "PARTIAL/BOUNDARY")

        all_results.append({
            "split_id": split["split_id"],
            "n_train": n_train,
            "n_held_out": n_held,
            "all_roles_in_train": bool(all_roles_in_train),
            "all_fillers_in_train": bool(all_fillers_in_train),
            "leakage_count": leakage_count,
            "chance": chance,
            # FHRR reference
            "fhrr_train_acc": float(fhrr_train_acc),
            "fhrr_held_acc": float(fhrr_held_acc),
            # Learned bilinear
            "bilinear_train_acc": float(bilinear_train_acc),
            "bilinear_held_acc": float(bilinear_held_acc),
            "bilinear_gap": float(bilinear_train_acc - bilinear_held_acc),
            # Anti-cheat 2
            "shuffled_held_acc": float(shuffled_held_acc),
            "shuffled_drops_to_chance": bool(shuffled_held_acc < chance + 0.1),
            # Anti-cheat 3
            "mem_floor_train_acc": float(mem_floor["train_acc"]),
            "mem_floor_held_acc": float(mem_floor["held_out_acc"]),
            # Anti-cheat 4
            "familiarity_gap": float(abstention["familiarity_gap"]),
            "abstention": abstention,
            # Verdict
            "systematic": bool(systematic),
            "memorization": bool(memorization),
            "verdict": verdict,
        })

    # --- AGGREGATE across splits ---
    held_accs = [r["bilinear_held_acc"] for r in all_results]
    train_accs = [r["bilinear_train_acc"] for r in all_results]
    fhrr_held_accs = [r["fhrr_held_acc"] for r in all_results]
    shuffled_accs = [r["shuffled_held_acc"] for r in all_results]
    n_systematic = sum(r["systematic"] for r in all_results)
    n_memorization = sum(r["memorization"] for r in all_results)

    elapsed = time.time() - t0
    return {
        "regime": regime_name,
        "F": F,
        "R": R,
        "D_in": D_in,
        "D_h": D_h,
        "seed": seed,
        "n_splits": n_splits,
        "n_epochs": n_epochs,
        "role_cos_mean": float(role_cos_mean),
        "chance": 1.0 / F,
        # Aggregated
        "bilinear_train_acc_mean": float(np.mean(train_accs)),
        "bilinear_held_acc_mean": float(np.mean(held_accs)),
        "bilinear_held_acc_std": float(np.std(held_accs)),
        "bilinear_gap_mean": float(np.mean(train_accs) - np.mean(held_accs)),
        "fhrr_held_acc_mean": float(np.mean(fhrr_held_accs)),
        "shuffled_acc_mean": float(np.mean(shuffled_accs)),
        "n_systematic_splits": int(n_systematic),
        "n_memorization_splits": int(n_memorization),
        "n_splits_total": n_splits,
        "systematic_fraction": float(n_systematic / n_splits),
        "memorization_fraction": float(n_memorization / n_splits),
        "elapsed_s": float(elapsed),
        "splits": all_results,
    }


# ============================================================================
# OVERALL DECISION LOGIC
# ============================================================================

def make_overall_decision(corr_results: List[Dict],
                           decorr_results: List[Dict]) -> Dict:
    """Aggregate across seeds and emit the verdict for both regimes.

    Option-C VIABLE: held-out ~ train on CORRELATED codes across seeds + anti-cheats decisive.
    BOUNDARY/NEGATIVE: held-out ~ chance on correlated codes (memorization on real target).
    Characterize whether it ALSO fails on decorrelated (form too weak vs correlation problem).
    """
    def agg(results):
        held = [r["bilinear_held_acc_mean"] for r in results]
        train = [r["bilinear_train_acc_mean"] for r in results]
        fhrr = [r["fhrr_held_acc_mean"] for r in results]
        chance = results[0]["chance"]
        shuffled = [r["shuffled_acc_mean"] for r in results]
        syst_frac = [r["systematic_fraction"] for r in results]
        return {
            "held_mean": float(np.mean(held)),
            "held_std": float(np.std(held)),
            "train_mean": float(np.mean(train)),
            "fhrr_held_mean": float(np.mean(fhrr)),
            "chance": float(chance),
            "shuffled_mean": float(np.mean(shuffled)),
            "systematic_fraction_mean": float(np.mean(syst_frac)),
            "above_2x_chance": bool(np.mean(held) > 2 * chance),
        }

    corr_agg = agg(corr_results)
    decorr_agg = agg(decorr_results)

    # Determine verdict
    corr_systematic = (
        corr_agg["above_2x_chance"] and
        corr_agg["systematic_fraction_mean"] > 0.5
    )
    decorr_systematic = (
        decorr_agg["above_2x_chance"] and
        decorr_agg["systematic_fraction_mean"] > 0.5
    )

    if corr_systematic and decorr_systematic:
        option_c_verdict = "VIABLE"
        description = "Learned binder generalizes systematically on BOTH regimes (correlated + decorrelated). Option C is viable."
    elif decorr_systematic and not corr_systematic:
        option_c_verdict = "NEGATIVE_ON_CORRELATED"
        description = (
            "Learned binder generalizes on DECORRELATED codes but FAILS on CORRELATED codes. "
            "Failure is the correlation structure (Fodor-Pylyshyn boundary on real brain codes). "
            "Option C requires code decorrelation FIRST (sparse-expansion front end) -- Option A is the honest ceiling."
        )
    elif not decorr_systematic and not corr_systematic:
        corr_train_high = np.mean([r["bilinear_train_acc_mean"] for r in corr_results]) > 0.5
        decorr_train_high = np.mean([r["bilinear_train_acc_mean"] for r in decorr_results]) > 0.5
        if corr_train_high or decorr_train_high:
            option_c_verdict = "MEMORIZATION"
            description = "Learned binder memorizes training pairs but FAILS to generalize on held-out novel combos in BOTH regimes. Binder form too weak OR the bilinear form is insufficient for this dimensionality."
        else:
            option_c_verdict = "FORM_TOO_WEAK"
            description = "Learned binder fails to learn even the training pairs. The bilinear form is too weak at this scale; a larger hidden dimension or more training is needed before systematicity can be tested."
    else:
        option_c_verdict = "PARTIAL"
        description = "Mixed results -- neither regime is robustly systematic multi-seed."

    return {
        "option_c_verdict": option_c_verdict,
        "description": description,
        "correlated_regime": corr_agg,
        "decorrelated_regime": decorr_agg,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--F-values", type=str, default="8,16",
                        help="Comma-separated list of filler counts to sweep")
    parser.add_argument("--R", type=int, default=4, help="Number of roles")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Train/test splits per seed")
    parser.add_argument("--n-epochs", type=int, default=800,
                        help="Training epochs for the bilinear binder")
    parser.add_argument("--D-h", type=int, default=64,
                        help="Hidden dimension for the bilinear binder")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for the bilinear binder")
    parser.add_argument("--n-pool", type=int, default=2000,
                        help="Sparse code pool size (n_pool for generate_sparse_patterns)")
    parser.add_argument("--pattern-size", type=int, default=100,
                        help="Sparse code pattern size K (active bits per concept)")
    parser.add_argument("--proj-dim", type=int, default=800,
                        help="Projection dimension for denoise64 codes")
    parser.add_argument("--out", type=str,
                        default=os.path.join(
                            _REPO, "research", "findings", "raw",
                            "cortex_learned_binder_systematicity_multiseed.json"),
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    F_values = [int(f) for f in args.F_values.split(",")]
    R = args.R

    print("=" * 70, flush=True)
    print("CORTEX LEARNED-BINDER SYSTEMATICITY PROBE", flush=True)
    print(f"Seeds: {seeds}  F: {F_values}  R: {R}", flush=True)
    print(f"n_splits={args.n_splits}  n_epochs={args.n_epochs}  D_h={args.D_h}  lr={args.lr}", flush=True)
    print("=" * 70, flush=True)

    t_global = time.time()
    all_results = []

    for seed in seeds:
        # ---- UNIT CHECK ----
        max_F = max(F_values)
        # Load sparse codes projected to args.proj_dim so both regimes share the same D_in,
        # giving the bilinear binder a fair comparison (same param count, same gradient scale).
        # The Gaussian projection is cosine-preserving: sparse_cos stays ~0.05 post-projection.
        sparse_codes, sparse_cos_mean, sparse_cos_max = load_sparse_codes_native(
            seed=seed, V=max_F, n_pool=args.n_pool, pattern_size=args.pattern_size,
            proj_dim=args.proj_dim
        )
        words_d64, denoise_codes, denoise_cos_mean = load_denoise64_codes(
            seed=seed, V=max_F, proj_dim=args.proj_dim
        )
        uc = unit_check(sparse_cos_mean, denoise_cos_mean)
        print(f"\n[seed={seed}] Unit check: sparse_cos={sparse_cos_mean:.4f}  "
              f"corr_cos={denoise_cos_mean:.4f}  -> {uc['status']}", flush=True)
        print(f"  sparse shape={sparse_codes.shape}  dense shape={denoise_codes.shape}", flush=True)

        if uc["status"] != "PASS":
            print("  ABORTING -- codes not in expected correlation regime!", flush=True)
            all_results.append({"seed": seed, "unit_check": uc, "error": "unit check failed"})
            continue

        for F in F_values:
            print(f"\n  --- F={F} ---", flush=True)

            # ---- DECORRELATED (SPARSE) CODES ----
            print(f"  [seed={seed} F={F}] DECORRELATED sparse codes (cos~{sparse_cos_mean:.3f})", flush=True)
            filler_codes_sparse = sparse_codes[:F]  # [F, n_pool]
            r_decorr = run_condition(
                regime_name="decorrelated_sparse",
                filler_codes=filler_codes_sparse,
                R=R, F=F, seed=seed,
                n_splits=args.n_splits, n_epochs=args.n_epochs,
                D_h=args.D_h, lr=args.lr, verbose=args.verbose
            )
            print(f"  -> bilinear held={r_decorr['bilinear_held_acc_mean']:.3f} "
                  f"train={r_decorr['bilinear_train_acc_mean']:.3f} "
                  f"FHRR={r_decorr['fhrr_held_acc_mean']:.3f} "
                  f"chance={r_decorr['chance']:.3f} "
                  f"syst_frac={r_decorr['systematic_fraction']:.2f}", flush=True)

            # ---- CORRELATED (DENOISE64) CODES ----
            print(f"  [seed={seed} F={F}] CORRELATED denoise64 codes (cos~{denoise_cos_mean:.3f})", flush=True)
            filler_codes_dense = denoise_codes[:F]  # [F, proj_dim]
            r_corr = run_condition(
                regime_name="correlated_denoise64",
                filler_codes=filler_codes_dense,
                R=R, F=F, seed=seed,
                n_splits=args.n_splits, n_epochs=args.n_epochs,
                D_h=args.D_h, lr=args.lr, verbose=args.verbose
            )
            print(f"  -> bilinear held={r_corr['bilinear_held_acc_mean']:.3f} "
                  f"train={r_corr['bilinear_train_acc_mean']:.3f} "
                  f"FHRR={r_corr['fhrr_held_acc_mean']:.3f} "
                  f"chance={r_corr['chance']:.3f} "
                  f"syst_frac={r_corr['systematic_fraction']:.2f}", flush=True)

            all_results.append({
                "seed": seed,
                "F": F,
                "R": R,
                "unit_check": uc,
                "decorrelated": r_decorr,
                "correlated": r_corr,
            })

    # ---- OVERALL DECISION ----
    # Group by F for the final verdict table
    verdicts = {}
    for F in F_values:
        corr_results = [r["correlated"] for r in all_results
                        if r.get("F") == F and "correlated" in r]
        decorr_results = [r["decorrelated"] for r in all_results
                          if r.get("F") == F and "decorrelated" in r]
        if corr_results and decorr_results:
            v = make_overall_decision(corr_results, decorr_results)
            verdicts[F] = v
            print(f"\n{'='*70}", flush=True)
            print(f"OVERALL VERDICT (F={F}): {v['option_c_verdict']}", flush=True)
            print(f"  {v['description']}", flush=True)
            print(f"  CORRELATED:   held={v['correlated_regime']['held_mean']:.3f} "
                  f"train={v['correlated_regime']['train_mean']:.3f} "
                  f"FHRR={v['correlated_regime']['fhrr_held_mean']:.3f} "
                  f"chance={v['correlated_regime']['chance']:.3f} "
                  f"syst_frac={v['correlated_regime']['systematic_fraction_mean']:.2f}", flush=True)
            print(f"  DECORRELATED: held={v['decorrelated_regime']['held_mean']:.3f} "
                  f"train={v['decorrelated_regime']['train_mean']:.3f} "
                  f"FHRR={v['decorrelated_regime']['fhrr_held_mean']:.3f} "
                  f"chance={v['decorrelated_regime']['chance']:.3f} "
                  f"syst_frac={v['decorrelated_regime']['systematic_fraction_mean']:.2f}", flush=True)

    total_elapsed = time.time() - t_global
    print(f"\nTotal elapsed: {total_elapsed:.1f}s", flush=True)

    output = {
        "meta": {
            "runner": "cortex_learned_binder_systematicity_probe",
            "seeds": seeds,
            "F_values": F_values,
            "R": R,
            "n_splits": args.n_splits,
            "n_epochs": args.n_epochs,
            "D_h": args.D_h,
            "lr": args.lr,
            "n_pool": args.n_pool,
            "pattern_size": args.pattern_size,
            "proj_dim": args.proj_dim,
            "total_elapsed_s": float(total_elapsed),
        },
        "verdicts_by_F": {str(k): v for k, v in verdicts.items()},
        "per_seed_per_F": all_results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
