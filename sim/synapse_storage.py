"""SSD-backed sparse synapse paging — Phase 3 of the CPU/RAM/SSD tiering design.

Lets pathways spill from RAM to NVMe when dormant, reload on demand
when activity rises. Like ZFS L2ARC: a warm tier sitting between hot
RAM and cold archive storage.

Pathway = the unit of paging. Each pathway has its own CSR sparse
matrix and tracks how recently it was accessed. Configurable eviction
policy (idle-steps threshold).

Persistence format: NumPy .npz (built-in, self-contained). One shard
file per pathway. CSR triplets (data, indices, indptr) plus shape
metadata. Compressed. allow_pickle=False on load (pure-array format
only — no arbitrary code execution risk).

Design doc: docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md (Phase 3)

Usage (simplified):

    from sim.synapse_storage import TieredSynapseStore

    store = TieredSynapseStore(root="bridges/synapse_shards/main")
    store.add_pathway("language_input_to_motor_N", csr_matrix)
    store.add_pathway("motor_N_to_language_output", csr_matrix)

    # Per simulation step
    fired = compute_fired_pathways()  # set of names that fired
    store.step(fired)  # bumps idle counters; evicts dormant pathways

    # Access (transparent: loads from disk if paged out)
    M = store.get_pathway("language_input_to_motor_N")
    # M is a scipy.sparse.csr_matrix; pathway now back in RAM
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default eviction policy: 1000 idle simulation steps before pageout.
# At 0.5 ms/step on 16w arch, that's ~500 ms of inactivity — long enough
# to avoid thrash on borderline-active pathways, short enough that
# dormant pathways don't squat in RAM forever.
DEFAULT_EVICT_AFTER_IDLE_STEPS = 1000

# Hysteresis on the page-in side: after a pathway is reloaded, give it
# a grace period before it can be evicted again. Prevents oscillation.
DEFAULT_GRACE_AFTER_PAGEIN_STEPS = 100


@dataclass
class PathwayShard:
    """Sparse-synapse shard backed by an NVMe file.

    A shard has TWO possible states:
    - in_memory=True: cached_csr holds a scipy.sparse.csr_matrix
    - in_memory=False: cached_csr is None; the .npz on disk is the source

    All accesses route through TieredSynapseStore.get_pathway() which
    handles the page-in transparently.
    """
    pathway_name: str
    shard_path: Path
    in_memory: bool = False
    cached_csr: Any = None  # scipy.sparse.csr_matrix when in_memory


class TieredSynapseStore:
    """RAM + SSD tier for sparse synapses, paged by pathway-activity.

    Policy:
    - New pathways added via add_pathway() start in RAM.
    - Each call to step(fired_pathways) bumps the idle counter for
      pathways NOT in fired_pathways, and resets to 0 for those that did.
    - Pathways with idle_count > evict_after_idle_steps get paged out
      (CSR -> .npz on disk; in-memory cache cleared).
    - Accessing a paged-out pathway loads the .npz back; idle counter
      and grace period reset.

    Lineage integration: TieredSynapseStore.save_all_shards(dir) and
    .load_shard_index(dir) handle the save/load boundary. The lineage
    system calls save_all_shards before its atomic h5 save.
    """

    def __init__(self,
                 root: Path | str,
                 evict_after_idle_steps: int = DEFAULT_EVICT_AFTER_IDLE_STEPS,
                 grace_after_pagein_steps: int = DEFAULT_GRACE_AFTER_PAGEIN_STEPS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.evict_after_idle_steps = int(evict_after_idle_steps)
        self.grace_after_pagein_steps = int(grace_after_pagein_steps)

        # Pathway state
        self.shards: dict[str, PathwayShard] = {}
        self.idle_counter: dict[str, int] = {}
        # Grace remaining (resets on page-in; decrements each step)
        self.grace_remaining: dict[str, int] = {}

        # Stats for inspection / metrics
        self.n_pageins: int = 0
        self.n_pageouts: int = 0

    # ── Pathway lifecycle ────────────────────────────────────────────

    def add_pathway(self, name: str, csr_matrix) -> None:
        """Register a new pathway, starting in RAM.

        Args:
            name: pathway identifier (e.g. "language_input_to_motor_N")
            csr_matrix: scipy.sparse.csr_matrix to track
        """
        if name in self.shards:
            raise ValueError(f"Pathway '{name}' already registered")
        shard = PathwayShard(
            pathway_name=name,
            shard_path=self.root / f"{name}.npz",
            in_memory=True,
            cached_csr=csr_matrix,
        )
        self.shards[name] = shard
        self.idle_counter[name] = 0
        self.grace_remaining[name] = 0

    def has_pathway(self, name: str) -> bool:
        return name in self.shards

    def get_pathway(self, name: str):
        """Return the CSR matrix for a pathway. Pages in if needed.

        Returns:
            scipy.sparse.csr_matrix (always; loads from disk if necessary)

        Raises:
            KeyError: if pathway not registered.
        """
        if name not in self.shards:
            raise KeyError(f"Pathway '{name}' not registered")
        shard = self.shards[name]
        if not shard.in_memory:
            self._page_in(shard)
        # Reset idle: pathway accessed
        self.idle_counter[name] = 0
        return shard.cached_csr

    def pathway_names(self) -> list[str]:
        """List of all registered pathway names (in registration order)."""
        return list(self.shards.keys())

    # ── Step / activity policy ────────────────────────────────────────

    def step(self, fired_pathways: set[str]) -> dict[str, str]:
        """Tick the activity policy. Returns dict of {name: action} for any
        pathways whose state changed this step.

        Args:
            fired_pathways: set of pathway names that fired this step.

        Returns:
            Dict mapping pathway name -> "evicted" for any that got paged
            out. Empty dict if no state changes. Useful for logging.
        """
        actions = {}
        for name in list(self.shards.keys()):
            if name in fired_pathways:
                self.idle_counter[name] = 0
            else:
                self.idle_counter[name] = self.idle_counter.get(name, 0) + 1
            # Decay grace
            if self.grace_remaining.get(name, 0) > 0:
                self.grace_remaining[name] -= 1

            # Eviction check (only if pathway is in-memory, idle past
            # threshold, and grace period has expired)
            shard = self.shards[name]
            if (shard.in_memory
                    and self.idle_counter[name] > self.evict_after_idle_steps
                    and self.grace_remaining.get(name, 0) == 0):
                self._page_out(shard)
                actions[name] = "evicted"
        return actions

    # ── Page in / page out (atomic-write for safety) ──────────────────

    def _page_in(self, shard: PathwayShard) -> None:
        """Load a shard from disk into RAM.

        Uses np.load with allow_pickle=False — pure-array format only,
        no arbitrary code execution path.
        """
        import scipy.sparse as sp
        import numpy as np

        if not shard.shard_path.exists():
            raise FileNotFoundError(
                f"Shard file missing on page-in: {shard.shard_path}"
            )
        # allow_pickle=False is the SAFE option — refuses any non-array
        # content. Our shards only contain numpy arrays + scalars.
        with np.load(shard.shard_path, allow_pickle=False) as data:
            csr = sp.csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=(int(data["n_post"]), int(data["n_pre"])),
            )
        shard.cached_csr = csr
        shard.in_memory = True
        self.idle_counter[shard.pathway_name] = 0
        self.grace_remaining[shard.pathway_name] = self.grace_after_pagein_steps
        self.n_pageins += 1

    def _page_out(self, shard: PathwayShard) -> None:
        """Persist a shard to disk; release RAM. Atomic write."""
        self._write_shard_in_place(shard)
        # Release RAM
        shard.cached_csr = None
        shard.in_memory = False
        self.n_pageouts += 1

    # ── Lineage save / load (full snapshot, all-in-memory) ────────────

    def save_all_shards(self) -> int:
        """Page in any dormant pathways, then write ALL shards to disk.

        Used by the lineage system at save time to ensure a consistent
        snapshot. Returns the number of shards written.
        """
        # First, page in everything that's currently dormant so we can
        # dump a consistent snapshot
        for name, shard in self.shards.items():
            if not shard.in_memory:
                self._page_in(shard)
        # Now write all shards as if evicting (they stay in memory)
        n = 0
        for name, shard in self.shards.items():
            self._write_shard_in_place(shard)
            n += 1
        # Write an index manifest so we can list pathways without scanning
        manifest = {
            "pathways": list(self.shards.keys()),
            "n_pageins": self.n_pageins,
            "n_pageouts": self.n_pageouts,
        }
        (self.root / "_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return n

    def _write_shard_in_place(self, shard: PathwayShard) -> None:
        """Write shard contents to disk WITHOUT evicting from RAM.

        Used by save_all_shards: shard stays in memory but is also
        persisted. Atomic write via .new + os.replace.

        Gotcha: np.savez_compressed auto-appends ".npz" to whatever
        filename we pass. To produce <name>.npz.new we pass <name>.npz_new
        (no extension) and numpy creates <name>.npz_new.npz; we then
        rename that to <name>.npz.
        """
        import numpy as np

        csr = shard.cached_csr
        # Pass a name without .npz so numpy doesn't double-suffix
        tmp_base = shard.shard_path.with_suffix(".npz_new")  # e.g. p.npz_new
        np.savez_compressed(
            tmp_base,  # numpy will append .npz -> p.npz_new.npz
            data=csr.data,
            indices=csr.indices,
            indptr=csr.indptr,
            n_post=np.array(csr.shape[0]),
            n_pre=np.array(csr.shape[1]),
            pathway_name=np.array(shard.pathway_name),
        )
        # numpy created <tmp_base>.npz (i.e. p.npz_new.npz)
        actual_tmp = Path(str(tmp_base) + ".npz")
        if not actual_tmp.exists():
            raise RuntimeError(
                f"savez_compressed didn't produce expected file: {actual_tmp}"
            )
        # Atomic rename to target
        if shard.shard_path.exists():
            os.replace(str(actual_tmp), str(shard.shard_path))
        else:
            actual_tmp.rename(shard.shard_path)

    def load_shard_index(self) -> int:
        """Discover pathways from existing .npz shards in self.root.

        Registers each as in_memory=False (will page-in on first access).
        Used by the lineage system at load time. Returns number of
        pathways registered.
        """
        manifest_path = self.root / "_manifest.json"
        pathway_names = []
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                pathway_names = manifest.get("pathways", [])
            except (json.JSONDecodeError, OSError):
                pass
        # Fallback: scan directory for .npz files
        if not pathway_names:
            pathway_names = [
                p.stem for p in self.root.glob("*.npz")
            ]
        for name in pathway_names:
            shard_path = self.root / f"{name}.npz"
            if not shard_path.exists():
                continue
            if name in self.shards:
                continue  # already registered
            self.shards[name] = PathwayShard(
                pathway_name=name,
                shard_path=shard_path,
                in_memory=False,
                cached_csr=None,
            )
            self.idle_counter[name] = 0
            self.grace_remaining[name] = 0
        return len(pathway_names)

    # ── Inspection / metrics ─────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot of store state — for logging + tests + webapp."""
        in_mem = sum(1 for s in self.shards.values() if s.in_memory)
        on_disk = len(self.shards) - in_mem
        return {
            "n_pathways": len(self.shards),
            "n_in_memory": in_mem,
            "n_on_disk": on_disk,
            "n_pageins_lifetime": self.n_pageins,
            "n_pageouts_lifetime": self.n_pageouts,
            "evict_after_idle_steps": self.evict_after_idle_steps,
            "grace_after_pagein_steps": self.grace_after_pagein_steps,
        }
