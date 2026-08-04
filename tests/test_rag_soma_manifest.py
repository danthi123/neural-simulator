from pathlib import Path
import sys
import types

# The engine test environment intentionally does not install LlamaIndex. These
# tests target the updater's pure manifest/batching helpers; live updater
# verification uses the dedicated RAG interpreter.
_build_stub = types.ModuleType("build_llamaindex_full")
_build_stub.RAG_ROOT = "/tmp/test-rag"
_build_stub.PERSIST = "/tmp/test-rag/llamaindex_full"
_build_stub.SIM = "/tmp/test-sim"
_build_stub.SOURCES = []
_build_stub.EXCLUDE_BASENAMES = set()
sys.modules.setdefault("build_llamaindex_full", _build_stub)

from tools.rag import update_indexes


def test_legacy_soma_manifest_is_rebuilt_before_incremental_migration():
    legacy = {
        r"E:\Documents\Projects\sim\research\findings\old.md": {
            "mtime": 1,
            "ids": ["old-node"],
        }
    }
    assert update_indexes._soma_manifest_needs_rebuild(legacy) is True


def test_stable_soma_manifest_survives_worktree_path_changes(tmp_path: Path):
    current = {
        "sim:research/findings/result.md": {
            "path": str(tmp_path / "sim-worktrees" / "topic" / "research/findings/result.md"),
            "mtime": 1,
            "ids": ["node-1"],
        },
        "catalog:feature-catalog.md": {
            "path": str(tmp_path / "sim-catalog" / "references" / "feature-catalog.md"),
            "mtime": 2,
            "ids": ["node-2"],
        },
    }
    assert update_indexes._soma_manifest_needs_rebuild(current) is False


def test_soma_file_storage_uses_bounded_batch_api(monkeypatch, tmp_path: Path):
    class Chunk:
        def __init__(self, number):
            self.text = f"chunk {number}"
            self.path = "result.md"
            self.heading = f"h{number}"

    class Memory:
        def __init__(self):
            self.calls = []

        def store_batch(self, texts, *, metadatas):
            self.calls.append((texts, metadatas))
            return [f"node-{text.split()[-1]}" for text in texts]

        def store(self, *args, **kwargs):  # pragma: no cover - catches regression
            raise AssertionError("single-record SOMA store used during refresh")

    monkeypatch.setattr(update_indexes, "SOMA_BATCH_SIZE", 2)
    memory = Memory()
    source = tmp_path / "result.md"
    source.write_text("ignored", encoding="utf-8")

    ids = update_indexes._store_file(
        memory,
        lambda text, path: [Chunk(i) for i in range(3)],
        lambda path: path.read_text(encoding="utf-8"),
        str(source),
    )

    assert ids == ["node-0", "node-1", "node-2"]
    assert [len(texts) for texts, _ in memory.calls] == [2, 1]


def test_host_heavy_lease_defers_overlapping_maintenance(monkeypatch, tmp_path):
    monkeypatch.setattr(update_indexes, "HOST_HEAVY_LEASE", str(tmp_path / "host.lock"))
    first = update_indexes.acquire_host_heavy_lease()
    assert first is not None
    try:
        assert update_indexes.acquire_host_heavy_lease() is None
    finally:
        first.close()
