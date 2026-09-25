"""Regression test for the independent re-review finding (2026-09-25, slotbinder-fast-teach-final FINAL fix
round, HIGH):

The MEDIUM-1 fix in tests/test_sparse_gain_index_sets_self_heals.py (compare the gain array's CONTENTS against a
cached snapshot on every call, `cp.array_equal` over all nnz) was CORRECT but made `_sparse_gain_index_sets`
redo exactly the O(nnz) work `cfg.sparse_activity_step` exists to avoid, on EVERY call -- twice per step where
anything fired -- silently defeating the flag's own speedup at scale (the whole point was O(fired) instead of
O(nnz); an O(nnz) staleness check on every call puts the O(nnz) cost right back).

The fix (sim/bridge.py, this commit): `cp_plasticity_rate_gain` is always stored as a `_TrackedGainArray` (a
`cp.ndarray` subclass whose own `__setitem__` bumps a per-instance `_mutation_version` counter), and
`_sparse_gain_index_sets` keys its cache on `(id(array), nnz, mutation_version)` -- three O(1) reads, no content
scan -- rebuilding the index sets only when one of those actually changed.

This test pins the O(1) property directly: repeated calls to `_sparse_gain_index_sets()` with NO intervening
write must (a) never invoke `cp.array_equal` (the specific O(nnz) primitive the prior fix used), and (b) return
the SAME cached array objects (proving no rebuild happened at all, not merely that one particular O(nnz) API
went unused). Both assertions FAIL against the pre-fix (content-diffing) implementation in
sim/bridge.py@bf661ed9c, which calls `cp.array_equal` on every single call regardless of staleness.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._keystone2_spiking_slot_binder_derisk import build_binder_bridge  # noqa: E402
from research.runners.slotbinder_composer import SlotBinderComposer  # noqa: E402

_VOCAB = ["dog", "cat", "fish", "bird", "chase", "eat", "see", "hear"]


def _small_binder_bridge():
    # K=2/KF=2: enough for a real, non-trivial cp_plasticity_rate_gain array without paying for a full
    # teach/retrieve run -- this test is about the CACHE's cost, not the composer's semantics.
    return build_binder_bridge(11, K=2, KF=2, sparse_step=True)


def test_repeated_calls_do_not_rescan_array_contents(monkeypatch):
    """No intervening write between calls -> zero calls to cp.array_equal (the O(nnz) primitive the prior,
    over-eager fix used on every call) and the exact same cached index-set objects returned each time."""
    b = _small_binder_bridge()
    nnz = b.cp_connections.nnz
    g = b.cp_plasticity_rate_gain
    assert g is not None and nnz > 0
    g[:nnz] = 1.0  # one real write to warm the cache from a known state

    import sim.bridge as bridge_mod
    real_array_equal = bridge_mod.cp.array_equal
    calls = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_array_equal(*args, **kwargs)

    monkeypatch.setattr(bridge_mod.cp, "array_equal", _spy)

    nz0, pos0 = b._sparse_gain_index_sets()  # may legitimately rebuild once (first call after the write above)
    calls.clear()

    for _ in range(5):
        nz_i, pos_i = b._sparse_gain_index_sets()
        assert nz_i is nz0 and pos_i is pos0, (
            "cache rebuilt on a call with no intervening write -- staleness detection is not O(1) "
            "(a rebuild here means every call pays the O(nnz) cache-rebuild cost, exactly what this flag "
            "exists to avoid)")

    assert calls == [], (
        f"_sparse_gain_index_sets called cp.array_equal {len(calls)} time(s) across 5 no-op calls -- staleness "
        "is being detected via an O(nnz) content comparison, not the O(1) mutation-version check")


def test_mutation_version_is_the_staleness_signal_not_content_diffing():
    """A write DOES invalidate the cache (still correct -- see test_sparse_gain_index_sets_self_heals.py), and
    the array's own _mutation_version counter is what changed, not a separate content snapshot."""
    b = _small_binder_bridge()
    nnz = b.cp_connections.nnz
    g = b.cp_plasticity_rate_gain
    g[:nnz] = 1.0
    b._sparse_gain_index_sets()
    v0 = getattr(g, "_mutation_version", None)
    assert v0 is not None, "cp_plasticity_rate_gain is not a _TrackedGainArray -- the O(1) staleness check has no signal"

    g[0] = 0.0  # in-place write, no setter call
    v1 = getattr(g, "_mutation_version", None)
    assert v1 == v0 + 1, "an indexed write to cp_plasticity_rate_gain must bump _mutation_version by exactly 1"

    nz, pos = b._sparse_gain_index_sets()
    assert 0 not in set(int(i) for i in nz), "the write was not picked up despite the version bump"


def test_whole_array_reassignment_is_wrapped_and_detected():
    """bridge.cp_plasticity_rate_gain = <new array> (the ~40-runner whole-array-reassignment pattern) must be
    auto-wrapped as a _TrackedGainArray by the property setter, and treated as a fresh object (cache invalidated
    by identity) even if it coincidentally has the same nnz and a freshly-reset version=0."""
    import numpy as np
    b = _small_binder_bridge()
    nnz = b.cp_connections.nnz
    b.cp_plasticity_rate_gain[:nnz] = 1.0
    nz0, pos0 = b._sparse_gain_index_sets()
    assert nz0.size == nnz

    # Whole-array replacement with a PLAIN (untracked) array of all-zeros, same nnz.
    b.cp_plasticity_rate_gain = np.zeros(nnz, dtype=np.float32)
    assert type(b.cp_plasticity_rate_gain).__name__ == "_TrackedGainArray", (
        "a plain array assigned to cp_plasticity_rate_gain must be auto-wrapped by the property setter")

    nz1, pos1 = b._sparse_gain_index_sets()
    assert nz1.size == 0, (
        "STALE CACHE after whole-array reassignment: the new (all-zero) array was aliased against the OLD "
        "cache because identity was not checked")
    assert pos1.size == 0


def test_default_dense_path_is_byte_identical_regardless_of_gain_array_wrapping():
    """PARITY / MUTATION-CHECK (required by the FINAL fix round): cp_plasticity_rate_gain is now ALWAYS a
    _TrackedGainArray (the property setter wraps unconditionally, whether or not cfg.sparse_activity_step is
    ever turned on), so this must not perturb the UNCHANGED, DEFAULT (flag-OFF) dense step that every
    non-slotbinder caller in this repo runs -- e.g. via numpy/cupy ufunc subclass-output propagation changing an
    intermediate array's type where the dense Hebbian path reads `cp_plasticity_rate_gain[active_synapse_indices_heb]`
    (sim/bridge.py).

    Builds two composers at the SAME seed (cfg.seed, the only thing that actually seeds the substrate --
    `build_binder_bridge`/`SlotBinderComposer` already set it correctly; see the "actual_seed_used seeds
    nothing" trap in CLAUDE.md/docs/ENGINE_REFERENCE.md), forces ONE bridge's gain array back to a PLAIN
    (untracked) `cp.ndarray` immediately after build -- writing the SAME values directly to the
    `cp_plasticity_rate_gain` property's backing attribute, bypassing the setter, to reproduce the pre-fix
    representation -- and confirms a real teach+query run through the dense step (`sparse_step=False`, the
    default) is BIT-IDENTICAL either way: same stored weights, same answers. MUTATION-CHECK: this test is only
    meaningful if it can tell the two representations apart, so it first asserts the two bridges really do start
    with different gain-array TYPES (one tracked, one plain) before running anything."""
    import numpy as np
    from sim.backend import to_host

    def _run(force_plain_gain, seed=13):
        c = SlotBinderComposer(seed=seed, vocab=list(_VOCAB), max_facts=4, sparse_step=False)
        c._ensure()
        b = c._b
        if force_plain_gain:
            plain = np.asarray(to_host(b.cp_plasticity_rate_gain)).copy()
            b.__dict__["_cp_plasticity_rate_gain_arr"] = plain  # bypass the property setter's wrapping
        return c, b

    c_tracked, b_tracked = _run(False)
    c_plain, b_plain = _run(True)
    assert type(b_tracked.cp_plasticity_rate_gain).__name__ == "_TrackedGainArray"
    assert type(b_plain.cp_plasticity_rate_gain).__name__ != "_TrackedGainArray", (
        "test setup bug: force_plain_gain did not actually produce a plain array -- this test would pass "
        "vacuously (unable to distinguish the two representations)")
    assert b_tracked.core_config.sparse_activity_step is False and b_plain.core_config.sparse_activity_step is False

    facts = [("dog", "chase", "cat"), ("bird", "eat", "fish")]
    for agent, action, patient in facts:
        assert c_tracked.store(agent, action, patient) is True
        assert c_plain.store(agent, action, patient) is True
    answers_tracked = [c_tracked.query_patient(a, v) for a, v, _ in facts]
    answers_plain = [c_plain.query_patient(a, v) for a, v, _ in facts]

    assert answers_tracked == answers_plain, "the gain array's TYPE changed the dense step's answers"
    w_tracked = np.asarray(to_host(b_tracked.cp_connections.data))
    w_plain = np.asarray(to_host(b_plain.cp_connections.data))
    assert np.array_equal(w_tracked, w_plain), (
        "the gain array's TYPE (_TrackedGainArray vs plain ndarray) changed the dense step's stored weights -- "
        "the default (flag-off) path is not byte-identical")
