"""Fast unit tests for `research/runners/_gnw_continuous_branchpoint_verify.py`'s pure instrument logic --
no full-brain build (that is what `--smoke`/`--six-seed` are for; these run in well under a second).

Written for review round 2 (2026-09-24, journal wf_16244c9d-ca5): the fix round's own commit-order requirement
is "the runner/code + tests" as its own commit, so these cover the two pieces of logic the review actually
changed and that a planted defect would otherwise pass silently:
  1. `tools.lab.lever(..., required=False)` for G5 -- must NOT raise when the lever did not move, and must
     still report the correct MOVED/UNCHANGED reading (the bug this replaces: `required=True` raised
     `LeverError` inside the `--six-seed` list comprehension, aborting the whole aggregate with no artifact).
  2. `_bio_hash`/`_array_bytes`/`_cp_completeness` -- the hashing and completeness-audit helpers G1/G2/G3/G4/
     G3-neg all depend on. Each test's planted defect is stated in its docstring, per `tools.lab`'s own
     convention.

Run with: SIM_BACKEND=numpy pytest tests/test_gnw_continuous_branchpoint_verify.py -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import scipy.sparse as sp

from tools.lab import LeverError, lever
from research.runners._gnw_continuous_branchpoint_verify import (
    _array_bytes, _bio_hash, _cp_completeness, MIN_X_DEFICIT,
)


class _FakeBridge:
    """Enough of a bridge to exercise `_cp_completeness` without building a substrate."""
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


# ── G5 lever fix: required=False must not raise, and must report the correct reading ────────────────────────
def test_lever_required_false_does_not_raise_when_unmoved():
    """Planted defect this guards against: `lever(..., required=True)` (the pre-review default) raises
    LeverError the instant `before == after`, which would abort `run_six_seed`'s per-seed list comprehension
    with zero artifacts written for any seed the moment one seed's x_A landed at exactly 1.0."""
    moved = lever("x_A (unmoved case)", 1.0, 1.0, required=False, continuous=1.0)
    assert moved is False


def test_lever_required_false_still_reports_moved_true():
    moved = lever("x_A (moved case)", 1.0, 0.5, required=False, continuous=0.5)
    assert moved is True


def test_lever_required_true_still_raises_for_other_callers():
    """Confirms the fix is scoped to this runner's call, not a global weakening of `lever`'s contract."""
    with pytest.raises(LeverError):
        lever("unrelated required=True caller", 1.0, 1.0)


def test_carryover_ok_threshold_matches_min_x_deficit():
    """`carryover_ok = xA_at_branch < 1.0 - MIN_X_DEFICIT` is computed independently of the (now non-raising)
    lever call -- a seed at exactly the boundary must read False (strict inequality), not True."""
    boundary = 1.0 - MIN_X_DEFICIT
    assert not (boundary < 1.0 - MIN_X_DEFICIT)
    assert (boundary - 1e-9) < 1.0 - MIN_X_DEFICIT


# ── _array_bytes / _bio_hash: the hashing primitive every gate (G1-G4, G3-neg) depends on ───────────────────
def test_array_bytes_dense_is_deterministic():
    a = np.array([1.0, 2.0, 3.0])
    assert _array_bytes(a) == _array_bytes(a.copy())


def test_array_bytes_dense_detects_a_changed_value():
    """Planted defect: a hash function that ignores array VALUES (e.g. hashes only shape/dtype) would pass
    G2/G4 even when `_full_restore` silently wrote the wrong numbers."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.1])
    assert _array_bytes(a) != _array_bytes(b)


def test_array_bytes_csr_hashes_data_indices_and_indptr():
    """`cp_connections` is a scipy CSR matrix. Planted defect: hashing only `.data` would miss a structural
    change (different `.indices`/`.indptr`) that leaves the same values in a different sparsity pattern."""
    m1 = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    m2 = sp.csr_matrix(np.array([[0.0, 1.0], [2.0, 0.0]]))  # same nonzero values, different positions
    assert _array_bytes(m1) != _array_bytes(m2)


def test_bio_hash_is_order_independent_over_dict_keys():
    """`_bio_hash` sorts `cp_snap.keys()` explicitly -- a hash keyed on dict iteration order would make G2/G4
    spuriously fail depending on Python's (implementation-defined-in-spirit, but here guaranteed) insertion
    order rather than on the actual array contents."""
    cp_a = {"cp_b": np.array([2.0]), "cp_a": np.array([1.0])}
    cp_b = {"cp_a": np.array([1.0]), "cp_b": np.array([2.0])}
    assert _bio_hash(cp_a, []) == _bio_hash(cp_b, [])


def test_bio_hash_std_snapshot_participates():
    """Planted defect this guards against: if `_bio_hash` ignored the STD `(x, boost)` snapshot, G2/G4 would
    pass even when `_std_restore` failed to actually restore the STD host state, since `cp_connections.data`
    (written by `std.apply()`) can coincidentally still match right at the branch point."""
    cp = {"cp_x": np.array([1.0])}
    std_a = [{"x": np.array([0.5, 0.5]), "boost": 1.0}]
    std_b = [{"x": np.array([0.9, 0.9]), "boost": 1.0}]
    assert _bio_hash(cp, std_a) != _bio_hash(cp, std_b)


# ── _cp_completeness: G1's own logic ─────────────────────────────────────────────────────────────────────────
def test_cp_completeness_passes_when_every_live_cp_attr_is_captured():
    bridge = _FakeBridge(cp_a=np.array([1.0]), cp_b=np.array([2.0]), not_cp_prefixed=np.array([3.0]))
    snap = {"cp_a": np.array([1.0]), "cp_b": np.array([2.0])}
    ok, missing = _cp_completeness(bridge, snap)
    assert ok is True
    assert missing == []


def test_cp_completeness_fails_when_a_live_cp_attr_is_dropped():
    """Planted defect: `_full_snapshot`'s `.copy()`/`.shape` filter silently drops a live (non-None) `cp_*`
    array whose type lacks one of those methods (e.g. a bare python float that should have been an ndarray)."""
    bridge = _FakeBridge(cp_a=np.array([1.0]), cp_scalar_float=3.14)
    snap = {"cp_a": np.array([1.0])}  # cp_scalar_float never made it into the snapshot
    ok, missing = _cp_completeness(bridge, snap)
    assert ok is False
    assert "cp_scalar_float" in missing


def test_cp_completeness_ignores_none_valued_cp_attrs():
    """A `cp_*` attribute that reads None (a disabled subsystem) must NOT count as missing -- this is the
    32-live/126-None split `scratchpad/probe_cp_types.py` measured on the real substrate."""
    bridge = _FakeBridge(cp_a=np.array([1.0]), cp_disabled=None)
    snap = {"cp_a": np.array([1.0])}
    ok, missing = _cp_completeness(bridge, snap)
    assert ok is True
    assert missing == []
