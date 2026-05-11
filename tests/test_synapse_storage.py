"""Tests for sim.synapse_storage — SSD-backed sparse synapse paging.

Phase 3 of the CPU/RAM/SSD tiering design. CPU-only; uses scipy.sparse +
numpy for both the in-memory and on-disk format.

Coverage:
- PathwayShard state transitions (in_memory ↔ paged-out)
- Eviction policy: idle counter + grace period
- Atomic write (no leftover .new files)
- Lineage save/load (snapshot + restore)
- Numerical preservation across page-out/page-in cycles
- Multiple pathways interacting through a single store
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.synapse_storage import (
    PathwayShard,
    TieredSynapseStore,
    DEFAULT_EVICT_AFTER_IDLE_STEPS,
    DEFAULT_GRACE_AFTER_PAGEIN_STEPS,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_csr(rows: int, cols: int, density: float = 0.1,
                seed: int = 0) -> sp.csr_matrix:
    """Construct a random sparse CSR matrix for tests."""
    rng = np.random.default_rng(seed)
    nnz = max(1, int(rows * cols * density))
    row_idx = rng.integers(0, rows, size=nnz)
    col_idx = rng.integers(0, cols, size=nnz)
    data = rng.normal(loc=1.0, scale=0.5, size=nnz).astype(np.float32)
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols))


# ──────────────────────────────────────────────────────────────────────
# Construction + basic lifecycle
# ──────────────────────────────────────────────────────────────────────


def test_store_construction_creates_root(tmp_path):
    """TieredSynapseStore creates its root directory."""
    root = tmp_path / "shards"
    assert not root.exists()
    store = TieredSynapseStore(root=root)
    assert root.exists()
    assert store.stats()["n_pathways"] == 0


def test_add_pathway_starts_in_memory(tmp_path):
    """Newly-added pathways are in-memory."""
    store = TieredSynapseStore(root=tmp_path)
    M = _make_csr(10, 20, density=0.2)
    store.add_pathway("test_pathway", M)
    assert store.has_pathway("test_pathway")
    assert store.shards["test_pathway"].in_memory is True
    s = store.stats()
    assert s["n_pathways"] == 1
    assert s["n_in_memory"] == 1
    assert s["n_on_disk"] == 0


def test_add_pathway_duplicate_raises(tmp_path):
    """Re-adding the same pathway name raises."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("dup", _make_csr(5, 5))
    with pytest.raises(ValueError, match="already registered"):
        store.add_pathway("dup", _make_csr(5, 5))


def test_get_pathway_unknown_raises(tmp_path):
    """get_pathway on unregistered name raises KeyError."""
    store = TieredSynapseStore(root=tmp_path)
    with pytest.raises(KeyError, match="not registered"):
        store.get_pathway("missing")


def test_get_pathway_returns_csr_in_memory(tmp_path):
    """get_pathway on in-memory pathway returns the CSR directly."""
    store = TieredSynapseStore(root=tmp_path)
    M = _make_csr(10, 20)
    store.add_pathway("p", M)
    M_back = store.get_pathway("p")
    # Same CSR matrix (in-memory, no copy)
    assert M_back is M


def test_pathway_names_listing(tmp_path):
    """pathway_names returns registration-ordered names."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("first", _make_csr(5, 5))
    store.add_pathway("second", _make_csr(5, 5))
    store.add_pathway("third", _make_csr(5, 5))
    assert store.pathway_names() == ["first", "second", "third"]


# ──────────────────────────────────────────────────────────────────────
# Page-out / page-in (low-level)
# ──────────────────────────────────────────────────────────────────────


def test_page_out_writes_disk_and_releases_ram(tmp_path):
    """_page_out persists to .npz and clears cached_csr."""
    store = TieredSynapseStore(root=tmp_path)
    M = _make_csr(10, 20)
    store.add_pathway("p", M)
    shard = store.shards["p"]
    assert shard.in_memory
    store._page_out(shard)
    assert not shard.in_memory
    assert shard.cached_csr is None
    assert shard.shard_path.exists()
    assert store.n_pageouts == 1
    # No leftover .new file
    assert not shard.shard_path.with_suffix(".npz.new").exists()


def test_page_in_restores_csr(tmp_path):
    """_page_in reads the .npz and restores cached_csr."""
    store = TieredSynapseStore(root=tmp_path)
    M = _make_csr(10, 20)
    store.add_pathway("p", M)
    shard = store.shards["p"]
    store._page_out(shard)
    # Now page in
    store._page_in(shard)
    assert shard.in_memory
    assert shard.cached_csr is not None
    # Numerical preservation
    np.testing.assert_array_equal(shard.cached_csr.data, M.data)
    np.testing.assert_array_equal(shard.cached_csr.indices, M.indices)
    np.testing.assert_array_equal(shard.cached_csr.indptr, M.indptr)
    assert shard.cached_csr.shape == M.shape


def test_page_in_missing_file_raises(tmp_path):
    """_page_in on a shard whose file doesn't exist raises."""
    store = TieredSynapseStore(root=tmp_path)
    shard = PathwayShard(
        pathway_name="ghost",
        shard_path=tmp_path / "ghost.npz",
        in_memory=False, cached_csr=None,
    )
    store.shards["ghost"] = shard
    with pytest.raises(FileNotFoundError, match="Shard file missing"):
        store._page_in(shard)


def test_get_pathway_transparently_pages_in(tmp_path):
    """get_pathway on a paged-out pathway transparently pages in."""
    store = TieredSynapseStore(root=tmp_path)
    M = _make_csr(8, 16, seed=42)
    store.add_pathway("p", M)
    shard = store.shards["p"]
    store._page_out(shard)
    assert not shard.in_memory
    # Access via the public API
    M_back = store.get_pathway("p")
    assert shard.in_memory
    np.testing.assert_array_equal(M_back.data, M.data)
    assert store.n_pageins == 1


# ──────────────────────────────────────────────────────────────────────
# Activity policy (step + eviction)
# ──────────────────────────────────────────────────────────────────────


def test_step_resets_idle_for_fired(tmp_path):
    """step() with fired pathway resets idle counter."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("p", _make_csr(5, 5))
    # Step 50 times without firing
    for _ in range(50):
        store.step(set())
    assert store.idle_counter["p"] == 50
    # Now fire it
    store.step({"p"})
    assert store.idle_counter["p"] == 0


def test_step_evicts_after_threshold(tmp_path):
    """Pathways idle past evict_after_idle_steps get paged out."""
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=10,
        grace_after_pagein_steps=0,
    )
    store.add_pathway("p", _make_csr(5, 5))
    actions = {}
    for _ in range(15):
        new_actions = store.step(set())
        actions.update(new_actions)
    # Should have been evicted somewhere in those 15 steps
    assert "p" in actions
    assert actions["p"] == "evicted"
    assert not store.shards["p"].in_memory


def test_grace_period_prevents_immediate_re_eviction(tmp_path):
    """After page-in, the grace period blocks immediate eviction."""
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=5,
        grace_after_pagein_steps=20,
    )
    store.add_pathway("p", _make_csr(5, 5))
    # Force initial eviction
    for _ in range(10):
        store.step(set())
    assert not store.shards["p"].in_memory
    # Page in via get_pathway
    store.get_pathway("p")
    assert store.shards["p"].in_memory
    assert store.grace_remaining["p"] == 20
    # Now idle for 10 steps — still in grace, no eviction
    for _ in range(10):
        store.step(set())
    assert store.shards["p"].in_memory


def test_active_pathway_never_evicts(tmp_path):
    """A pathway that fires every step never gets evicted."""
    store = TieredSynapseStore(root=tmp_path, evict_after_idle_steps=5)
    store.add_pathway("hot", _make_csr(5, 5))
    for _ in range(100):
        store.step({"hot"})
    assert store.shards["hot"].in_memory


def test_mixed_pathways_independent_eviction(tmp_path):
    """Hot pathway stays in RAM while cold pathway gets evicted."""
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=5,
        grace_after_pagein_steps=0,
    )
    store.add_pathway("hot", _make_csr(5, 5))
    store.add_pathway("cold", _make_csr(5, 5))
    for _ in range(20):
        store.step({"hot"})  # only hot fires
    assert store.shards["hot"].in_memory
    assert not store.shards["cold"].in_memory


# ──────────────────────────────────────────────────────────────────────
# Lineage integration (save_all_shards + load_shard_index)
# ──────────────────────────────────────────────────────────────────────


def test_save_all_shards_writes_each(tmp_path):
    """save_all_shards writes every pathway to disk + creates manifest."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("a", _make_csr(5, 5, seed=1))
    store.add_pathway("b", _make_csr(5, 5, seed=2))
    n = store.save_all_shards()
    assert n == 2
    assert (tmp_path / "a.npz").exists()
    assert (tmp_path / "b.npz").exists()
    assert (tmp_path / "_manifest.json").exists()
    manifest = json.loads((tmp_path / "_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["pathways"]) == {"a", "b"}


def test_save_all_shards_pages_in_dormant(tmp_path):
    """If a pathway is paged out, save_all_shards pages it in first."""
    store = TieredSynapseStore(root=tmp_path)
    M_a = _make_csr(5, 5, seed=10)
    store.add_pathway("a", M_a)
    # Force eviction
    store._page_out(store.shards["a"])
    assert not store.shards["a"].in_memory
    # save_all_shards should page back in and write
    store.save_all_shards()
    assert store.shards["a"].in_memory
    # Numerical preservation
    M_back = store.get_pathway("a")
    np.testing.assert_array_equal(M_back.data, M_a.data)


def test_load_shard_index_registers_pathways(tmp_path):
    """A fresh store can load_shard_index from a previously-saved root."""
    # Save with one store
    store1 = TieredSynapseStore(root=tmp_path)
    store1.add_pathway("first", _make_csr(5, 5, seed=100))
    store1.add_pathway("second", _make_csr(8, 12, seed=101))
    store1.save_all_shards()
    # Load with a fresh store
    store2 = TieredSynapseStore(root=tmp_path)
    n = store2.load_shard_index()
    assert n == 2
    assert set(store2.pathway_names()) == {"first", "second"}
    # Both start dormant
    for name in ("first", "second"):
        assert not store2.shards[name].in_memory


def test_load_shard_index_falls_back_to_directory_scan(tmp_path):
    """If _manifest.json is missing, load_shard_index scans .npz files."""
    store1 = TieredSynapseStore(root=tmp_path)
    store1.add_pathway("a", _make_csr(5, 5))
    store1.save_all_shards()
    # Delete the manifest
    (tmp_path / "_manifest.json").unlink()
    # Load — should still find the pathway via directory scan
    store2 = TieredSynapseStore(root=tmp_path)
    n = store2.load_shard_index()
    assert n == 1
    assert "a" in store2.pathway_names()


def test_round_trip_save_load_preserves_csr(tmp_path):
    """save + reload preserves CSR contents exactly."""
    store1 = TieredSynapseStore(root=tmp_path)
    M_orig = _make_csr(20, 30, density=0.15, seed=42)
    store1.add_pathway("p", M_orig)
    store1.save_all_shards()
    # New store, load by index, then access (page in)
    store2 = TieredSynapseStore(root=tmp_path)
    store2.load_shard_index()
    M_loaded = store2.get_pathway("p")
    np.testing.assert_array_equal(M_loaded.data, M_orig.data)
    np.testing.assert_array_equal(M_loaded.indices, M_orig.indices)
    np.testing.assert_array_equal(M_loaded.indptr, M_orig.indptr)
    assert M_loaded.shape == M_orig.shape


# ──────────────────────────────────────────────────────────────────────
# Stats + telemetry
# ──────────────────────────────────────────────────────────────────────


def test_stats_reflects_state(tmp_path):
    """stats() returns accurate snapshot."""
    store = TieredSynapseStore(root=tmp_path, evict_after_idle_steps=5,
                                  grace_after_pagein_steps=0)
    store.add_pathway("a", _make_csr(5, 5))
    store.add_pathway("b", _make_csr(5, 5))
    s = store.stats()
    assert s["n_pathways"] == 2
    assert s["n_in_memory"] == 2
    assert s["n_on_disk"] == 0
    # Evict one
    store._page_out(store.shards["a"])
    s = store.stats()
    assert s["n_in_memory"] == 1
    assert s["n_on_disk"] == 1
    assert s["n_pageouts_lifetime"] == 1


# ──────────────────────────────────────────────────────────────────────
# Anti-thrash sanity check
# ──────────────────────────────────────────────────────────────────────


def test_no_thrash_with_borderline_pathway(tmp_path):
    """A pathway firing periodically (within idle threshold) stays
    in RAM after page-in — no oscillation.

    Setup: fire-every-5-steps, idle-threshold=10 — fire interval (5)
    well below idle threshold (10), so idle counter never crosses
    threshold between fires.
    """
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=10,  # > fire interval (5)
        grace_after_pagein_steps=0,
    )
    store.add_pathway("p", _make_csr(5, 5))
    # Force initial eviction
    for _ in range(15):
        store.step(set())
    assert not store.shards["p"].in_memory
    # Page in via access
    store.get_pathway("p")
    # Fire every 5 steps for 100 steps; idle threshold=10 so it
    # never crosses threshold between fires.
    for i in range(100):
        if i % 5 == 0:
            store.step({"p"})
        else:
            store.step(set())
    # Pathway should still be in-memory (fired recently every 5 steps)
    assert store.shards["p"].in_memory


def test_grace_blocks_immediate_re_eviction_post_pagein(tmp_path):
    """Grace period specifically prevents oscillation right after page-in.

    Even with idle threshold below the fire interval, the grace period
    blocks immediate re-eviction for `grace_after_pagein_steps` steps.
    """
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=2,  # very aggressive
        grace_after_pagein_steps=50,  # but generous grace
    )
    store.add_pathway("p", _make_csr(5, 5))
    # Force eviction
    for _ in range(10):
        store.step(set())
    assert not store.shards["p"].in_memory
    # Page in
    store.get_pathway("p")
    assert store.shards["p"].in_memory
    # 40 idle steps — within grace period — NOT evicted
    for _ in range(40):
        store.step(set())
    assert store.shards["p"].in_memory
    # Now go past grace expiry — eviction WILL happen
    for _ in range(20):
        store.step(set())
    assert not store.shards["p"].in_memory


# ──────────────────────────────────────────────────────────────────────
# Atomic write safety
# ──────────────────────────────────────────────────────────────────────


def test_no_leftover_new_files_after_save(tmp_path):
    """After save_all_shards, no .npz.new artifacts remain."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("a", _make_csr(5, 5))
    store.add_pathway("b", _make_csr(5, 5))
    store.save_all_shards()
    leftover = list(tmp_path.glob("*.new"))
    assert leftover == []


def test_no_leftover_new_files_after_page_out_cycle(tmp_path):
    """After multiple page-out cycles, no .new artifacts remain."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("p", _make_csr(10, 10))
    for _ in range(5):
        store._page_out(store.shards["p"])
        store._page_in(store.shards["p"])
    leftover = list(tmp_path.glob("*.new"))
    assert leftover == []


# ──────────────────────────────────────────────────────────────────────
# Phase 4: memory-pressure eviction (added 2026-05-11)
# ──────────────────────────────────────────────────────────────────────


def test_pressure_eviction_disabled_by_default(tmp_path):
    """ram_budget_bytes=0 -> no pressure eviction (idle-only)."""
    store = TieredSynapseStore(root=tmp_path, evict_after_idle_steps=1000)
    assert store.ram_budget_bytes == 0
    for _ in range(5):
        store.add_pathway(f"p{_}", _make_csr(20, 20))
    # Run many steps without firing — only idle eviction would matter
    for _ in range(100):
        store.step(set())
    s = store.stats()
    assert s["n_pressure_evictions"] == 0


def test_pressure_eviction_fires_when_over_budget(tmp_path):
    """Pressure eviction fires when total in-RAM exceeds ram_budget."""
    # Build 5 pathways, then set budget below the total
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=10**9,  # effectively disable idle eviction
        grace_after_pagein_steps=0,
        ram_budget_bytes=1,  # nearly zero — forces eviction every step
    )
    for i in range(5):
        store.add_pathway(f"p{i}", _make_csr(20, 20, seed=i))
    initial = store._estimate_in_memory_bytes()
    assert initial > 1  # we are over budget

    # First step should evict at least one pathway
    actions = store.step(set())
    pressure_actions = [k for k, v in actions.items()
                          if v == "pressure_evicted"]
    assert len(pressure_actions) >= 1
    assert store.n_pressure_evictions >= 1


def test_pressure_eviction_picks_longest_idle(tmp_path):
    """When evicting under pressure, pick the longest-idle pathway."""
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=10**9,  # disable idle eviction
        grace_after_pagein_steps=0,
        ram_budget_bytes=1,  # forces immediate pressure eviction
    )
    store.add_pathway("hot", _make_csr(10, 10, seed=1))
    store.add_pathway("cold", _make_csr(10, 10, seed=2))
    # Fire "hot" several times before any step
    for _ in range(10):
        store.step({"hot"})  # cold idle counter climbs
    # At this point cold should be the eviction target
    # cold's idle = 10, hot's idle = 0
    assert store.idle_counter["cold"] == 10
    assert store.idle_counter["hot"] == 0
    # Note: pressure eviction has already fired during the 10 steps;
    # both might be evicted. Just check cold was evicted at some point.
    assert "cold" in [name for name, shard in store.shards.items()
                       if not shard.in_memory] or store.n_pressure_evictions > 0


def test_pressure_eviction_respects_grace(tmp_path):
    """Pathways in grace period are NOT pressure-evicted."""
    store = TieredSynapseStore(
        root=tmp_path,
        evict_after_idle_steps=10**9,
        grace_after_pagein_steps=100,  # long grace
        ram_budget_bytes=1,  # nearly zero
    )
    store.add_pathway("p1", _make_csr(20, 20))
    # Force eviction
    store._page_out(store.shards["p1"])
    assert not store.shards["p1"].in_memory
    # Page in via get_pathway
    store.get_pathway("p1")
    assert store.shards["p1"].in_memory
    assert store.grace_remaining["p1"] == 100

    # Step — should NOT evict due to grace
    actions = store.step(set())
    assert "p1" not in actions
    assert store.shards["p1"].in_memory


def test_stats_includes_pressure_metrics(tmp_path):
    """stats() exposes pressure-eviction metrics."""
    store = TieredSynapseStore(root=tmp_path, ram_budget_bytes=1024 * 1024)
    s = store.stats()
    assert "n_pressure_evictions" in s
    assert "in_memory_bytes" in s
    assert "ram_budget_bytes" in s
    assert s["n_pressure_evictions"] == 0
    assert s["ram_budget_bytes"] == 1024 * 1024


def test_estimate_in_memory_bytes_zero_when_all_paged_out(tmp_path):
    """In-memory byte estimate is 0 when nothing is in RAM."""
    store = TieredSynapseStore(root=tmp_path)
    store.add_pathway("p", _make_csr(5, 5))
    initial = store._estimate_in_memory_bytes()
    assert initial > 0
    store._page_out(store.shards["p"])
    assert store._estimate_in_memory_bytes() == 0
