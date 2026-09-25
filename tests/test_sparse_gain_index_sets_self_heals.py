"""Regression test for the review finding (2026-09-25, slotbinder-fast-teach-final fix round, MEDIUM #1):

SimulationBridge._sparse_gain_index_sets (sim/bridge.py) cached the gain!=0 / gain>0 index sets keyed on
`(nnz, self._plasticity_gain_version)`. `_plasticity_gain_version` is bumped ONLY by set_plasticity_gate and
set_global_plasticity_gain (sim/bridge.py). About 40 research/runners sites write `cp_plasticity_rate_gain` IN
PLACE directly instead (`g[:] = 0.0`, `g[idx] = 1.0`, `g[:] = saved`), and for those callers the version never
bumps, so the cache silently served a STALE index set: `cfg.sparse_activity_step`'s gated decay/clip would then
touch the WRONG synapses (reproduced on a binder bridge with sparse_step on, K=6/KF=8/seed 11: freeze all gains
in place, run 40 steps, open one slot's gate in place, run 40 more -- 3,200 synapses differed from the dense
path). `_sparse_activity_step_can_dispatch` has no way to detect this (it only inspects config flags), so the
only protection was a warning in a sim/config.py comment.

The fix makes the cache SELF-HEALING: every call also compares the gain array's CONTENTS against the snapshot
taken when the cache was last built (one O(nnz) equality read) and rebuilds on any mismatch, version bump or
not. This test reproduces the exact bug pattern (an in-place write with no setter call) directly against a real
binder bridge and FAILS on the pre-fix cache (which trusts `(nnz, version)` alone).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners._keystone2_spiking_slot_binder_derisk import build_binder_bridge  # noqa: E402


def _small_binder_bridge():
    # K=2/KF=2 is enough to exercise a real, non-trivial cp_plasticity_rate_gain array without paying for a
    # full teach/retrieve run -- this test is about the CACHE, not the composer's semantics.
    return build_binder_bridge(11, K=2, KF=2, sparse_step=True)


def test_in_place_freeze_is_seen_without_a_version_bump():
    """The reviewer's exact repro, condensed: an in-place write that bypasses set_plasticity_gate entirely."""
    b = _small_binder_bridge()
    nnz = b.cp_connections.nnz
    g = b.cp_plasticity_rate_gain
    assert g is not None and g.shape[0] >= nnz > 0
    version_before = getattr(b, "_plasticity_gain_version", 0)

    # Warm the cache at all-ones (bypassing the setter, exactly the ~40-runner pattern under review).
    g[:nnz] = 1.0
    nz0, pos0 = b._sparse_gain_index_sets()
    assert nz0.size == nnz and pos0.size == nnz

    # IN-PLACE freeze: no setter call, so _plasticity_gain_version is untouched.
    g[:nnz] = 0.0
    assert getattr(b, "_plasticity_gain_version", 0) == version_before, (
        "test setup bug: an in-place write must not itself bump the version counter")

    nz1, pos1 = b._sparse_gain_index_sets()
    assert nz1.size == 0, (
        "STALE CACHE: _sparse_gain_index_sets still reports gain!=0 synapses after an in-place freeze that "
        "never bumped _plasticity_gain_version -- the gated Hebbian decay/clip would touch the wrong synapses")
    assert pos1.size == 0


def test_in_place_reopen_after_freeze_is_seen():
    """'open slot1's gate in place' half of the reviewer's repro: reopening a subset after a prior freeze."""
    b = _small_binder_bridge()
    nnz = b.cp_connections.nnz
    g = b.cp_plasticity_rate_gain

    g[:nnz] = 1.0
    b._sparse_gain_index_sets()  # warm the cache
    g[:nnz] = 0.0
    b._sparse_gain_index_sets()  # warm it again at all-frozen

    reopened = np.arange(min(5, nnz))
    g[reopened] = 1.0  # in-place, no setter
    nz, pos = b._sparse_gain_index_sets()
    assert sorted(np.asarray(nz).tolist()) == sorted(reopened.tolist()), (
        "STALE CACHE: reopened synapses are missing from the gain!=0 index set after an in-place write")
    assert sorted(np.asarray(pos).tolist()) == sorted(reopened.tolist())


def test_index_sets_always_match_the_current_array_contents():
    """Direct equivalence: after an arbitrary in-place write, the cached index sets must equal the boolean masks
    freshly derived from the array's CURRENT contents -- not from whatever was true when the cache was built."""
    b = _small_binder_bridge()
    nnz = b.cp_connections.nnz
    g = b.cp_plasticity_rate_gain

    g[:nnz] = 1.0
    b._sparse_gain_index_sets()  # warm cache at all-ones
    g[:nnz] = 0.0
    g[min(3, nnz - 1)] = 1.0  # in-place, single synapse reopened

    nz, pos = b._sparse_gain_index_sets()
    current = np.asarray(g[:nnz])
    expected_nz = np.flatnonzero(current != 0.0)
    expected_pos = np.flatnonzero(current > 0.0)
    assert sorted(np.asarray(nz).tolist()) == sorted(expected_nz.tolist())
    assert sorted(np.asarray(pos).tolist()) == sorted(expected_pos.tolist())
