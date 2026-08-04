#!/usr/bin/env python3
"""Fail-closed freshness check for the canonical scientific RAG index."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

try:
    from .rag_paths import resolve_paths
except ImportError:  # direct script execution
    from rag_paths import resolve_paths


EXCLUDE_BASENAMES = {"AUTONOMOUS_STATE.md", "AUTONOMOUS_STATE_ARCHIVE.md"}


def _patterns(repo: Path, catalog: Path) -> list[tuple[str, list[str]]]:
    return [
        ("finding", [str(repo / "research/findings/**/*.md")]),
        ("plan", [str(repo / "docs/plans/*.md")]),
        (
            "doc",
            [
                str(repo / "docs/*.md"),
                str(repo / "CLAUDE.md"),
                str(repo / "ROADMAP.md"),
                str(repo / "README.md"),
                str(repo / "GAP_CLOSURE_MISSION.md"),
                str(repo / "docs/FAILURE_GATE_MATRIX.md"),
            ],
        ),
        ("catalog", [str(catalog / "*.md")]),
        ("paper", [str(catalog / "textbooks/*/*.txt")]),
    ]


def corpus_manifest_hash(repo: Path, catalog: Path) -> str:
    """Match update_indexes.manifest_hash without importing the heavy RAG stack."""
    digest = hashlib.sha256()
    for _source_type, patterns in _patterns(repo, catalog):
        for pattern in patterns:
            for raw in sorted(glob.glob(pattern, recursive=True)):
                path = Path(raw)
                if path.name in EXCLUDE_BASENAMES:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                digest.update(str(path).encode("utf-8", "replace"))
                digest.update(str(int(stat.st_mtime)).encode())
                digest.update(str(stat.st_size).encode())
    return digest.hexdigest()


def inspect_index(repo: Path) -> dict[str, object]:
    paths = resolve_paths(repo)
    manifest_path = paths.rag_root / ".rag_manifest.json"
    schema_path = paths.rag_root / ".rag_schema.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "reason": f"index metadata unreadable: {exc}"}
    if schema.get("document_id_schema") != "repo-relative-v1":
        return {"status": "unavailable", "reason": "index schema is missing or unsupported"}
    source_repo = Path(schema.get("source_repo", "")).expanduser()
    if not source_repo.is_dir():
        return {"status": "unavailable", "reason": f"indexed source checkout is unavailable: {source_repo}"}
    if not paths.full_index.is_dir():
        return {"status": "unavailable", "reason": f"full index is unavailable: {paths.full_index}"}
    expected = manifest.get("hash")
    if not isinstance(expected, str) or len(expected) != 64:
        return {"status": "unavailable", "reason": "index manifest hash is malformed"}
    source_paths = resolve_paths(source_repo)
    actual = corpus_manifest_hash(source_repo.resolve(), source_paths.catalog)
    if actual != expected:
        return {
            "status": "stale",
            "reason": "indexed corpus differs from the current source/catalog manifest",
            "expected_hash": expected,
            "actual_hash": actual,
            "source_repo": str(source_repo.resolve()),
        }
    return {
        "status": "current",
        "manifest_hash": actual,
        "source_repo": str(source_repo.resolve()),
        "catalog": str(source_paths.catalog),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = inspect_index(Path.cwd())
    if args.json:
        json.dump(result, sys.stdout, sort_keys=True)
        print()
    else:
        print(f"RAG_INDEX_{str(result['status']).upper()}: {result.get('reason', '')}".rstrip())
    return 0 if result["status"] == "current" else 1


if __name__ == "__main__":
    raise SystemExit(main())
