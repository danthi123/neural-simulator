"""Regression gate (2026-09-05, rank-6 pickle bug): `ShardedPhasorStore.save()` raised `TypeError: cannot
pickle 'mappingproxy' object` whenever a shard's `kb` held per-fact SUBSTRATE-STORE handles
(`enable_substrate_store=True`) instead of plain numpy composite arrays.

ROOT CAUSE. `_store_substrate` (`research/runners/rf_phasor_composer.py`) returns a live `SimulationBridge`
object per fact -- the composite phase vector lives in the bridge's synaptic weights, not a numpy array.
`save()`'s original `comps.append(np.asarray(handle))` did not raise on that call (numpy silently wraps an
arbitrary object in a 0-d OBJECT array); the failure surfaced two steps later inside `np.savez`'s pickling of
the resulting object array, because `SimulationBridge` sets `snr_packet_bindings`/`snr_packet_kernel_parameters`/
`snr_packet_hh_phi` (`sim/bridge.py`) to `types.MappingProxyType(...)` instance attributes, and a mappingproxy
has no pickle support.

FIX (`research/runners/sharded_phasor_store.py`, no `sim/` edit). `save()` now reads the composite phase vector
back out via the composer's own `_retrieve_substrate` BEFORE it ever reaches numpy (the same call every
substrate-store query already makes), so `composites.npz` only ever holds plain real-valued arrays --
byte-identical on-disk shape/dtype to the numpy-kb path. `load()`'s mirror-image fix rebuilds the substrate
handle from that vector via `_store_substrate` (the identical call `RFPhasorComposer.store()` makes on first
write), deterministic given the manifest's seed.

This file is the gate `research/findings/2026-09-05-rank6-shardedphasorstore-pickle-fix-*.md` and
`research/FAILURE_LOG.md`'s 2026-09-05 row point to. See also the numpy-kb-path parity test in
`test_rf_phasor_composer_substrate_store_parity` (`tests/test_rf_phasor_composer.py`), which validates the
LIVE (in-memory, non-persisted) substrate-vs-numpy answer parity this file extends across a save/load
roundtrip.
"""
from __future__ import annotations

import os

import pytest

from research.runners.sharded_phasor_store import ShardedPhasorStore

# A cyclic little world (agent/action/patient triples) exercising query_patient (agent-cued), query_agent
# (the reverse fan-out lookup), ask_yes_no, and render_fact -- the store's full conversational API.
FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]


def _build(tmp_path_factory, seed, enable_substrate_store):
    store = ShardedPhasorStore(n_shards=4, seed=seed, D=128, period=200,
                                enable_substrate_store=enable_substrate_store)
    for a, v, p in FACTS:
        store.store(a, v, p)
    return store


def _assert_recall_identical(pre, post):
    """Every read the conversational API exposes must agree BYTE-FOR-BYTE (string/None equality -- these are
    discrete phase-cleanup answers, not floats) between the pre-save store and the post-load store, on both
    the stored facts (recall) and a genuinely-unstored cue (the no-confab moat)."""
    for a, v, p in FACTS:
        assert pre.query_patient(a, v) == post.query_patient(a, v) == p
        assert pre.query_agent(v, p) == post.query_agent(v, p) == a
        assert pre.render_fact(a) == post.render_fact(a)
        assert pre.ask_yes_no(a, v, p) == post.ask_yes_no(a, v, p)     # exact-triple read, whatever it is
    # the moat: a never-taught agent must abstain identically before and after the roundtrip.
    assert pre.query_patient("elephant", "go") is None
    assert post.query_patient("elephant", "go") is None
    assert pre.query_agent("go", "river") is None                      # river's action is "look", not "go"
    assert post.query_agent("go", "river") is None


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_save_load_roundtrip_numpy_kb_baseline(tmp_path_factory, seed):
    """The default (`enable_substrate_store=False`) path already worked; this is the byte-identical-behavior
    baseline the substrate-store test below is compared against -- a regression here would mean this fix
    changed the UNCHANGED path, which it must not (the task's own "no change to picklable-field semantics")."""
    store = _build(tmp_path_factory, seed, enable_substrate_store=False)
    path = str(tmp_path_factory.mktemp(f"ltm_numpy_{seed}") / "bundle")
    n = store.save(path)
    assert n == len(FACTS)
    loaded = ShardedPhasorStore.load(path)
    assert loaded.total_facts() == len(FACTS)
    _assert_recall_identical(store, loaded)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_save_load_roundtrip_substrate_store_survives_and_matches(tmp_path_factory, seed):
    """THE GATE. `enable_substrate_store=True` must (a) SAVE WITHOUT RAISING (before the fix: `TypeError:
    cannot pickle 'mappingproxy' object`, raised inside `np.savez`) and (b) answer every read identically
    before vs. after the save/load roundtrip -- the loaded store's rebuilt substrate handles must be
    functionally indistinguishable from the ones `store()` built directly."""
    store = _build(tmp_path_factory, seed, enable_substrate_store=True)
    path = str(tmp_path_factory.mktemp(f"ltm_substrate_{seed}") / "bundle")
    n = store.save(path)               # must not raise -- this line IS the regression gate
    assert n == len(FACTS)
    # save() must not leave a corrupt/partial bundle: all three files present and each individually loadable.
    for fname in ("manifest.json", "facts.json", "composites.npz"):
        assert os.path.isfile(os.path.join(path, fname)), f"save() did not produce {fname}"
    loaded = ShardedPhasorStore.load(path)
    assert loaded.total_facts() == len(FACTS)
    # the manifest's own composer_kwargs round-trips the flag, so the reloaded shards are ALSO substrate-store
    # composers (not silently downgraded to the numpy-kb path) -- confirms load() actually rebuilt handles via
    # `_store_substrate` rather than merely happening to answer correctly with the wrong representation.
    assert all(sh.enable_substrate_store for sh in loaded.shards)
    _assert_recall_identical(store, loaded)
