"""Bridge Lineage Manager — persistent continuous-learning state.

Per user (2026-05-10): "we're basically starting from scratch on each
run. Is there a good way to continually work off the most recently
trained sim state and keep improving it?"

A BridgeLineage is a git-like persistent history of a trained bridge's
state. Each lineage lives under `bridges/lineage/<name>/`:
  - current.simstate.h5     ← the latest state (auto-loaded)
  - metadata.json           ← vocab tier, training events, growth events
  - _growth_log.md          ← human-readable diary
  - history/                ← periodic snapshots

Two workflow modes coexist:
  - Science mode: --from-scratch, multi-seed reproducibility
  - Continuous mode: load lineage on start, save on exit; sim "lives"
    between sessions

Design doc: docs/plans/2026-05-10-bridge-lineage-design.md
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


LINEAGE_ROOT = Path("bridges/lineage")


# Schema version of metadata.json. Bump when format changes.
METADATA_SCHEMA_VERSION = 1


@dataclass
class GrowthEvent:
    """A single notable event in a lineage's history.

    Examples: init from scratch, tier promotion, structural growth event,
    significant accuracy milestone, fork point.
    """
    at: str  # ISO 8601 timestamp
    kind: str  # init / tier_promotion / structural_growth / milestone / fork
    description: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AccuracyDatapoint:
    """A single recorded accuracy measurement."""
    at: str
    metric: str  # e.g. "A2W any" / "W2A primary" / "retention_overall"
    value: float
    context: str = ""  # e.g. "post-training", "after_tier_promotion"


@dataclass
class LineageMetadata:
    """Persistent metadata for a lineage.

    Stored as metadata.json alongside the .simstate.h5 file.
    """
    lineage_name: str
    schema_version: int = METADATA_SCHEMA_VERSION
    created_at: str = ""
    last_updated_at: str = ""
    current_tier: str = "4-word"  # human-readable tier name
    vocab: list[str] = field(default_factory=list)
    arch: dict = field(default_factory=dict)
    cumulative_training_events: int = 0
    accuracy_history: list[dict] = field(default_factory=list)
    growth_events: list[dict] = field(default_factory=list)
    parent_lineage: str | None = None  # null for main; set for forks
    branched_at: str | None = None  # when forked (ISO 8601)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "LineageMetadata":
        """Tolerant loader: ignores unknown fields, fills missing with defaults."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)

    def add_growth_event(self, kind: str, description: str = "", **metadata):
        event = GrowthEvent(
            at=datetime.now().isoformat(timespec="seconds"),
            kind=kind,
            description=description,
            metadata=metadata,
        )
        self.growth_events.append(asdict(event))

    def add_accuracy(self, metric: str, value: float, context: str = ""):
        point = AccuracyDatapoint(
            at=datetime.now().isoformat(timespec="seconds"),
            metric=metric,
            value=value,
            context=context,
        )
        self.accuracy_history.append(asdict(point))


class BridgeLineage:
    """Persistent lineage of a trained bridge's state.

    Usage:
        lineage = BridgeLineage("main")
        if lineage.exists():
            bridge = lineage.load(load_fn=my_bridge_loader)
        else:
            bridge = train_bridge(...)
            lineage.save(bridge, save_fn=my_bridge_saver,
                          tier="8-word", arch={...})

        # ... use bridge ...

        # On session end, save back:
        lineage.save(bridge, save_fn=my_bridge_saver)

    File layout under bridges/lineage/<name>/:
        current.simstate.h5  - latest state (the loaded one)
        metadata.json        - persistent metadata
        _growth_log.md       - human-readable diary
        history/             - periodic snapshots
            <timestamp>-checkpoint.simstate.h5
            <timestamp>-checkpoint.metadata.json

    Atomic save: writes to .new file, fsync, atomic rename. No partial-
    write corruption.
    """

    def __init__(self, name: str = "main", root: Path = None):
        self.name = name
        self.root = (root or LINEAGE_ROOT) / name
        self.current_path = self.root / "current.simstate.h5"
        self.metadata_path = self.root / "metadata.json"
        self.history_dir = self.root / "history"
        self.growth_log_path = self.root / "_growth_log.md"

    # ── Existence checks ────────────────────────────────────────────────

    def exists(self) -> bool:
        """True if this lineage has at least one saved state."""
        return self.current_path.exists() and self.metadata_path.exists()

    # ── Metadata read/write ────────────────────────────────────────────

    def read_metadata(self) -> LineageMetadata:
        """Load metadata.json. Returns a default-fresh metadata if file
        missing (lineage not yet saved)."""
        if not self.metadata_path.exists():
            return LineageMetadata(
                lineage_name=self.name,
                created_at=datetime.now().isoformat(timespec="seconds"),
                last_updated_at=datetime.now().isoformat(timespec="seconds"),
            )
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return LineageMetadata.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(
                f"Failed to read lineage metadata at {self.metadata_path}: {e}"
            )

    def write_metadata(self, metadata: LineageMetadata):
        """Atomic write of metadata.json. Updates last_updated_at."""
        metadata.last_updated_at = datetime.now().isoformat(timespec="seconds")
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.metadata_path.with_suffix(".new")
        tmp_path.write_text(
            json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Atomic rename (Windows-compatible: must remove target first)
        if self.metadata_path.exists():
            os.replace(str(tmp_path), str(self.metadata_path))
        else:
            tmp_path.rename(self.metadata_path)

    # ── Bridge save/load ────────────────────────────────────────────────

    def save(self, bridge, save_fn=None,
             tier: str = None, arch: dict = None,
             metadata_updates: dict = None,
             snapshot: bool = True):
        """Atomically save bridge state to lineage.

        Args:
            bridge: SimulationBridge instance.
            save_fn: callable(bridge, path_str) that writes bridge state
                to the given path. If None, calls bridge.save_checkpoint(path).
            tier: optional tier label to update metadata.current_tier.
            arch: optional arch dict to update metadata.arch.
            metadata_updates: optional dict of fields to merge into metadata.
            snapshot: if True (default), also archive a history entry
                BEFORE overwriting current. The history entry is the
                PREVIOUS current.simstate.h5 (so history reflects
                last-saved state, not current).
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the previous state if it exists
        if snapshot and self.current_path.exists():
            self._snapshot_current_to_history()

        # Save bridge to .new then atomic rename
        tmp_path = self.current_path.with_suffix(".h5.new")
        if save_fn is None:
            bridge.save_checkpoint(str(tmp_path))
        else:
            save_fn(bridge, str(tmp_path))
        # Atomic rename
        if self.current_path.exists():
            os.replace(str(tmp_path), str(self.current_path))
        else:
            tmp_path.rename(self.current_path)

        # Update metadata
        meta = self.read_metadata()
        if tier is not None:
            meta.current_tier = tier
        if arch is not None:
            meta.arch.update(arch)
        if metadata_updates:
            for k, v in metadata_updates.items():
                setattr(meta, k, v)
        self.write_metadata(meta)

    def load(self, bridge_loader=None, mode: str = "synonym",
             seed: int = 42):
        """Load lineage state into a bridge.

        Args:
            bridge_loader: callable(checkpoint_path, mode, seed) that
                returns a SimulationBridge. If None, the caller must use
                read_metadata() + checkpoint_path to construct + load
                via their own logic (matches chat_repl._load_bridge_from_checkpoint).
            mode: bridge mode (tier1/synonym/synonym12/synonym16).
                Defaults to "synonym". Used by bridge_loader.
            seed: bridge seed. Defaults to 42.

        Returns:
            The bridge (if loader provided) or the checkpoint path string
            (so caller can build their own bridge).
        """
        if not self.exists():
            raise FileNotFoundError(
                f"Lineage '{self.name}' has no current state. "
                f"Expected {self.current_path}"
            )
        if bridge_loader is None:
            return str(self.current_path)
        return bridge_loader(str(self.current_path), mode, seed)

    # ── History management ─────────────────────────────────────────────

    def _snapshot_current_to_history(self):
        """Archive current.simstate.h5 to history/ with timestamp.

        Uses millisecond precision in the timestamp to avoid collisions
        on rapid-fire saves (e.g. in tests, or programmatic batch
        training).
        """
        if not self.current_path.exists():
            return
        # Millisecond precision avoids collisions during rapid saves
        now = datetime.now()
        ts = now.strftime("%Y-%m-%dT%H-%M-%S") + f"-{now.microsecond // 1000:03d}"
        history_path = self.history_dir / f"{ts}-checkpoint.simstate.h5"
        # If somehow still a collision (sub-millisecond saves), append a counter
        counter = 0
        while history_path.exists():
            counter += 1
            history_path = self.history_dir / (
                f"{ts}-{counter}-checkpoint.simstate.h5"
            )
        # Copy (not move) so current remains valid during the snapshot.
        # Then the save() flow replaces current atomically.
        shutil.copy2(str(self.current_path), str(history_path))
        # Also copy metadata at this point. Use a CONSISTENT naming
        # convention: <snap_id>-checkpoint.metadata.json — this is what
        # rollback_to() and the bridge_lineage CLI's diff command expect.
        # (Previously this used `.simstate.metadata.json` which created
        # orphaned metadata files unreadable by rollback / diff.)
        if self.metadata_path.exists():
            # snap_id is everything before "-checkpoint.simstate.h5"
            snap_id = history_path.name.replace("-checkpoint.simstate.h5", "")
            history_meta = self.history_dir / f"{snap_id}-checkpoint.metadata.json"
            shutil.copy2(str(self.metadata_path), str(history_meta))

    def list_history(self) -> list[Path]:
        """List history snapshots in chronological order."""
        if not self.history_dir.exists():
            return []
        return sorted(self.history_dir.glob("*-checkpoint.simstate.h5"))

    def rollback_to(self, snapshot_id: str):
        """Restore a history snapshot as the new current.

        Args:
            snapshot_id: timestamp string (e.g. "2026-05-10T22-00") matching
                a file in history/.
        """
        history_path = self.history_dir / f"{snapshot_id}-checkpoint.simstate.h5"
        history_meta = self.history_dir / f"{snapshot_id}-checkpoint.metadata.json"
        if not history_path.exists():
            raise FileNotFoundError(f"No history snapshot: {snapshot_id}")
        # First archive current state
        self._snapshot_current_to_history()
        # Then restore
        shutil.copy2(str(history_path), str(self.current_path))
        if history_meta.exists():
            shutil.copy2(str(history_meta), str(self.metadata_path))

    def prune_history(self, keep_last: int = 30):
        """Remove all but the most recent N history snapshots."""
        snapshots = self.list_history()
        if len(snapshots) <= keep_last:
            return
        for snap in snapshots[:-keep_last]:
            try:
                snap.unlink()
                # Also remove paired metadata file if present.
                # Naming convention: <snap_id>-checkpoint.metadata.json
                snap_id = snap.name.replace("-checkpoint.simstate.h5", "")
                paired_meta = self.history_dir / f"{snap_id}-checkpoint.metadata.json"
                if paired_meta.exists():
                    paired_meta.unlink()
            except OSError:
                pass  # best-effort; don't fail save on cleanup errors

    # ── Forking ────────────────────────────────────────────────────────

    def fork(self, new_name: str) -> "BridgeLineage":
        """Create a new lineage branched from this one's current state.

        The new lineage starts with current.simstate.h5 = this lineage's
        current state. Metadata records the parent and branch time.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"Cannot fork from non-existent lineage '{self.name}'"
            )
        new_lineage = BridgeLineage(new_name, root=self.root.parent)
        if new_lineage.exists():
            raise FileExistsError(
                f"Lineage '{new_name}' already exists; cannot fork."
            )
        new_lineage.root.mkdir(parents=True, exist_ok=True)
        new_lineage.history_dir.mkdir(parents=True, exist_ok=True)

        # Copy bridge state
        shutil.copy2(str(self.current_path), str(new_lineage.current_path))

        # Build new metadata derived from parent
        parent_meta = self.read_metadata()
        now = datetime.now().isoformat(timespec="seconds")
        new_meta = LineageMetadata(
            lineage_name=new_name,
            schema_version=METADATA_SCHEMA_VERSION,
            created_at=now,
            last_updated_at=now,
            current_tier=parent_meta.current_tier,
            vocab=list(parent_meta.vocab),
            arch=dict(parent_meta.arch),
            cumulative_training_events=parent_meta.cumulative_training_events,
            accuracy_history=list(parent_meta.accuracy_history),
            growth_events=list(parent_meta.growth_events),
            parent_lineage=self.name,
            branched_at=now,
            tags=[f"fork-of-{self.name}"],
        )
        new_meta.add_growth_event(
            kind="fork",
            description=f"Forked from lineage '{self.name}'",
            parent=self.name,
        )
        new_lineage.write_metadata(new_meta)
        return new_lineage

    # ── Per-pathway shard export (tiering Phase 3 Strategy C) ────────

    def export_shards(self, bridge, shard_root: Path | str = None) -> int:
        """Export the bridge's pathways as per-pathway shards.

        Strategy C of the tiering Phase 3 part 2 design:
        - Bridge stays monolithic (cp_connections unchanged)
        - Lineage gains a side-car shards/ directory with per-pathway
          .npz files alongside current.simstate.h5
        - Future use: SSD-tiered access via TieredSynapseStore

        Args:
            bridge: SimulationBridge with region_manager enabled
            shard_root: optional path to write shards into (default:
                self.root / "shards")

        Returns:
            Number of shards written (= number of pathways in
            region_manager.pathways()).

        Raises:
            RuntimeError: if bridge doesn't have a region_manager.
        """
        from sim.synapse_storage import TieredSynapseStore

        if bridge.region_manager is None:
            raise RuntimeError(
                "export_shards: bridge.region_manager is None — "
                "brain region framework must be enabled"
            )
        shard_root = Path(shard_root) if shard_root else (self.root / "shards")
        shard_root.mkdir(parents=True, exist_ok=True)
        store = TieredSynapseStore(root=shard_root)
        per_pathway = bridge.extract_per_pathway_csrs()
        for name, csr in per_pathway.items():
            store.add_pathway(name, csr)
        n = store.save_all_shards()
        return n

    def list_shards(self, shard_root: Path | str = None) -> list[str]:
        """List pathway shard names available on disk for this lineage.

        Returns the pathway names that have been exported via
        export_shards(). Empty list if shards directory doesn't exist
        or no shards have been saved.
        """
        from sim.synapse_storage import TieredSynapseStore

        shard_root = Path(shard_root) if shard_root else (self.root / "shards")
        if not shard_root.exists():
            return []
        store = TieredSynapseStore(root=shard_root)
        store.load_shard_index()
        return store.pathway_names()

    # ── Growth log (human-readable diary of the lineage's evolution) ────

    def render_growth_log(self) -> str:
        """Render the lineage's growth_events + accuracy_history as a
        human-readable markdown diary. Pure function; does not write to
        disk (caller can write to _growth_log.md if desired).
        """
        meta = self.read_metadata()
        lines = []
        lines.append(f"# Growth log — lineage `{meta.lineage_name}`")
        lines.append("")
        # Header summary
        lines.append(f"- **Current tier:** {meta.current_tier}")
        lines.append(f"- **Cumulative training events:** "
                       f"{meta.cumulative_training_events}")
        if meta.parent_lineage:
            lines.append(f"- **Parent lineage:** `{meta.parent_lineage}`")
            if meta.branched_at:
                lines.append(f"- **Branched at:** {meta.branched_at}")
        if meta.tags:
            lines.append(f"- **Tags:** {', '.join(meta.tags)}")
        if meta.created_at:
            lines.append(f"- **Created:** {meta.created_at}")
        if meta.last_updated_at:
            lines.append(f"- **Last updated:** {meta.last_updated_at}")
        lines.append("")
        # Growth events timeline
        if meta.growth_events:
            lines.append("## Growth events")
            lines.append("")
            for e in meta.growth_events:
                at = e.get("at", "?")
                kind = e.get("kind", "?")
                desc = e.get("description", "")
                lines.append(f"- **[{at}] `{kind}`** — {desc}")
                extra_meta = e.get("metadata", {})
                if extra_meta:
                    # Render extra metadata as a nested list
                    for k, v in extra_meta.items():
                        # Skip noisy / verbose fields if any (e.g. weight arrays)
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            lines.append(f"  - `{k}`: {v}")
                        elif isinstance(v, dict):
                            short = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                            lines.append(f"  - `{k}`: {{{short}}}")
                        else:
                            lines.append(f"  - `{k}`: {v!r}")
            lines.append("")
        else:
            lines.append("_No growth events recorded yet._")
            lines.append("")
        # Accuracy history (most-recent first, capped)
        if meta.accuracy_history:
            lines.append("## Accuracy history (most recent 20)")
            lines.append("")
            lines.append("| At | Metric | Value | Context |")
            lines.append("|----|--------|-------|---------|")
            for p in list(reversed(meta.accuracy_history))[:20]:
                at = p.get("at", "?")
                m = p.get("metric", "?")
                v = p.get("value", 0)
                ctx = p.get("context", "")
                lines.append(f"| {at} | {m} | {v:.3f} | {ctx} |")
            lines.append("")
        return "\n".join(lines)

    def write_growth_log(self) -> Path:
        """Render + write the growth log to `_growth_log.md`.

        Returns the written path. Atomic via .new + os.replace.
        """
        content = self.render_growth_log()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.growth_log_path.with_suffix(".md.new")
        tmp.write_text(content, encoding="utf-8")
        if self.growth_log_path.exists():
            os.replace(str(tmp), str(self.growth_log_path))
        else:
            tmp.rename(self.growth_log_path)
        return self.growth_log_path

    # ── Listing ─────────────────────────────────────────────────────────

    @classmethod
    def list_all(cls, root: Path = None) -> list["BridgeLineage"]:
        """Return all known lineages under bridges/lineage/."""
        root = root or LINEAGE_ROOT
        if not root.exists():
            return []
        return [
            cls(p.name, root=root)
            for p in root.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        ]
