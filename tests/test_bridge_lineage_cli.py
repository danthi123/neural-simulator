"""Tests for the bridge_lineage CLI runner.

CPU-only — uses a mock bridge that satisfies the save_checkpoint
interface without needing GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.lineage import BridgeLineage, LineageMetadata


class _MockBridge:
    """Mock that satisfies save_checkpoint interface for testing the CLI."""
    def __init__(self, content: str = "fake-state"):
        self.content = content

    def save_checkpoint(self, path: str):
        Path(path).write_text(self.content, encoding="utf-8")


def _seed_lineage(tmp_path: Path, name: str = "test",
                    content: str = "v1") -> BridgeLineage:
    """Create a lineage with one save + a growth event."""
    lineage = BridgeLineage(name, root=tmp_path)
    bridge = _MockBridge(content)
    lineage.save(bridge, tier="4-word",
                  arch={"mode": "tier1", "n_neurons": 1000})
    meta = lineage.read_metadata()
    meta.cumulative_training_events = 200
    meta.vocab = ["north", "east", "south", "west"]
    meta.add_growth_event(kind="init", description="Test init")
    meta.add_accuracy(metric="A2W any", value=0.5, context="post-train")
    lineage.write_metadata(meta)
    return lineage


def _run_cli(*cli_args, root: Path = None,
              env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the CLI in a subprocess. Returns CompletedProcess."""
    base_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env:
        base_env.update(env)
    args = [sys.executable, "-m", "research.runners.bridge_lineage"]
    if root is not None:
        args.extend(["--root", str(root)])
    args.extend(cli_args)
    return subprocess.run(
        args, capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=base_env,
    )


def test_list_empty(tmp_path):
    """list on an empty root prints a friendly message."""
    p = _run_cli("list", root=tmp_path)
    assert p.returncode == 0
    assert "no lineages found" in p.stdout


def test_list_after_seeding(tmp_path):
    """list shows the lineage after a save."""
    _seed_lineage(tmp_path, name="main")
    _seed_lineage(tmp_path, name="experiment_x")
    p = _run_cli("list", root=tmp_path)
    assert p.returncode == 0
    assert "main" in p.stdout
    assert "experiment_x" in p.stdout
    assert "4-word" in p.stdout
    assert "200 events" in p.stdout


def test_show_existing_lineage(tmp_path):
    """show prints metadata for an existing lineage."""
    _seed_lineage(tmp_path, name="main")
    p = _run_cli("show", "main", root=tmp_path)
    assert p.returncode == 0
    assert "Lineage: main" in p.stdout
    assert "Current tier:   4-word" in p.stdout
    assert "Cumulative events: 200" in p.stdout
    assert "north" in p.stdout  # vocab preview


def test_show_missing_lineage(tmp_path):
    """show on a missing lineage exits with error."""
    p = _run_cli("show", "nonexistent", root=tmp_path)
    assert p.returncode == 2
    assert "does not exist" in p.stderr


def test_history_empty(tmp_path):
    """history on a lineage with no snapshots is fine."""
    _seed_lineage(tmp_path, name="main")
    p = _run_cli("history", "main", root=tmp_path)
    assert p.returncode == 0
    assert "no history snapshots" in p.stdout


def test_history_lists_snapshots(tmp_path):
    """history lists prior snapshots after multiple saves."""
    lineage = _seed_lineage(tmp_path, name="main", content="v1")
    # Save again to push v1 to history
    lineage.save(_MockBridge("v2"), tier="4-word")
    p = _run_cli("history", "main", root=tmp_path)
    assert p.returncode == 0
    assert "1 total" in p.stdout
    assert "checkpoint.simstate.h5" not in p.stdout  # we strip the suffix


def test_fork_creates_child(tmp_path):
    """fork creates a new lineage from a parent."""
    _seed_lineage(tmp_path, name="main")
    p = _run_cli("fork", "main", "experiment_x", root=tmp_path)
    assert p.returncode == 0
    assert "main' -> 'experiment_x'" in p.stdout
    # Verify the new lineage exists
    child = BridgeLineage("experiment_x", root=tmp_path)
    assert child.exists()
    meta = child.read_metadata()
    assert meta.parent_lineage == "main"


def test_fork_existing_target_fails(tmp_path):
    """fork into an existing lineage errors cleanly."""
    _seed_lineage(tmp_path, name="main")
    _seed_lineage(tmp_path, name="experiment_x")
    p = _run_cli("fork", "main", "experiment_x", root=tmp_path)
    assert p.returncode == 2
    assert "already exists" in p.stderr


def test_prune_keeps_recent(tmp_path):
    """prune trims old snapshots to the keep-last threshold."""
    lineage = _seed_lineage(tmp_path, name="main", content="v0")
    for i in range(5):
        lineage.save(_MockBridge(f"v{i+1}"), tier="4-word")
    p = _run_cli("prune", "main", "--keep-last", "2", root=tmp_path)
    assert p.returncode == 0
    assert "kept last 2" in p.stdout
    snapshots = lineage.list_history()
    assert len(snapshots) == 2


def test_rollback_to_snapshot(tmp_path):
    """rollback restores a prior snapshot as the current state."""
    lineage = _seed_lineage(tmp_path, name="main", content="v1")
    lineage.save(_MockBridge("v2"), tier="4-word")
    lineage.save(_MockBridge("v3"), tier="4-word")
    snapshots = lineage.list_history()
    assert len(snapshots) == 2
    # Roll back to the oldest snapshot
    snap_id = snapshots[0].name.replace("-checkpoint.simstate.h5", "")
    p = _run_cli("rollback", "main", "--to", snap_id, root=tmp_path)
    assert p.returncode == 0
    # Current is now v1 (the oldest snapshot's content)
    assert lineage.current_path.read_text(encoding="utf-8") == "v1"
    # Growth event was appended
    meta = lineage.read_metadata()
    assert any(e["kind"] == "rollback" for e in meta.growth_events)


def test_rollback_missing_snapshot(tmp_path):
    """rollback with an unknown snapshot ID errors."""
    _seed_lineage(tmp_path, name="main")
    p = _run_cli("rollback", "main", "--to", "9999-01-01T00-00-00",
                  root=tmp_path)
    assert p.returncode == 2
    assert "No history snapshot" in p.stderr


def test_diff_no_changes(tmp_path):
    """diff current vs current shows no changes."""
    _seed_lineage(tmp_path, name="main")
    p = _run_cli("diff", "main", "--from", "current", "--to", "current",
                  root=tmp_path)
    assert p.returncode == 0
    assert "no metadata differences" in p.stdout


def test_diff_current_vs_history(tmp_path):
    """diff between current and a history snapshot shows accumulated changes."""
    # First save → tier="4-word", arch={mode:"tier1"} (history snapshot
    # records this as the metadata at that point).
    lineage = _seed_lineage(tmp_path, name="main", content="v1")
    # Second save: tier promotion to 8-word + arch change to synonym.
    # The pre-save snapshot picks up the previous metadata, so the diff
    # should show both the tier and arch differences.
    lineage.save(_MockBridge("v2"), tier="8-word",
                  arch={"mode": "synonym"})
    snapshots = lineage.list_history()
    snap_id = snapshots[0].name.replace("-checkpoint.simstate.h5", "")
    p = _run_cli("diff", "main", "--from", snap_id, "--to", "current",
                  root=tmp_path)
    assert p.returncode == 0, p.stderr
    # The arch dict diff should always show 'mode' changing
    assert "mode" in p.stdout and "tier1" in p.stdout and "synonym" in p.stdout
