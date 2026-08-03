#!/usr/bin/env python3
"""Check or install the repository-controlled RAG hook workflow."""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from pathlib import Path

try:
    from .rag_paths import resolve_paths
except ImportError:  # direct script execution
    from rag_paths import resolve_paths


EXPECTED_HOOKS_PATH = "tools/githooks"
EXPECTED_SCHEMA = "repo-relative-v1"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def workflow_status(repo: Path) -> list[tuple[str, bool, str]]:
    paths = resolve_paths(repo)
    configured = _git(repo, "config", "--get", "core.hooksPath")
    hook_dir = repo / EXPECTED_HOOKS_PATH
    schema_path = paths.rag_root / ".rag_schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8")).get(
            "document_id_schema"
        )
    except Exception:
        schema = None
    checks = [
        (
            "hooks-path",
            configured == EXPECTED_HOOKS_PATH,
            configured or "unset",
        ),
        (
            "pre-commit-executable",
            os.access(hook_dir / "pre-commit", os.X_OK),
            str(hook_dir / "pre-commit"),
        ),
        (
            "post-commit-executable",
            os.access(hook_dir / "post-commit", os.X_OK),
            str(hook_dir / "post-commit"),
        ),
        (
            "engine-python",
            os.access(paths.engine_python, os.X_OK),
            str(paths.engine_python),
        ),
        (
            "rag-python",
            os.access(paths.rag_python, os.X_OK),
            str(paths.rag_python),
        ),
        ("full-index", paths.full_index.is_dir(), str(paths.full_index)),
        ("catalog", paths.catalog.is_dir(), str(paths.catalog)),
        (
            "index-schema",
            schema == EXPECTED_SCHEMA,
            schema or "missing/legacy; run tools/rag/update_indexes.py --rebuild",
        ),
    ]
    return checks


def install(repo: Path) -> None:
    _git(repo, "config", "core.hooksPath", EXPECTED_HOOKS_PATH)
    for name in ("pre-commit", "post-commit"):
        path = repo / EXPECTED_HOOKS_PATH / name
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if args.install:
        install(repo)
    checks = workflow_status(repo)
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'BLOCKED'}] {name}: {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print("RAG_WORKFLOW_BLOCKED: " + ", ".join(failed))
        return 1
    print("RAG_WORKFLOW_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
