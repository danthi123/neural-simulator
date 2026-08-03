"""Canonical paths shared by the RAG search, build, update, and hook checks."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RagPaths:
    repo: Path
    common_repo: Path
    projects_root: Path
    rag_root: Path
    catalog: Path
    engine_python: Path
    rag_python: Path

    @property
    def full_index(self) -> Path:
        return self.rag_root / "llamaindex_full"

    @property
    def findings_index(self) -> Path:
        return self.rag_root / "llamaindex_findings"


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _git_common_dir(repo: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return _absolute(result.stdout.strip())


def resolve_paths(
    repo: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    common_dir: str | os.PathLike[str] | None = None,
) -> RagPaths:
    """Resolve shared assets through Git's common checkout, not a worktree parent."""
    values = os.environ if env is None else env
    checkout = _absolute(
        values.get("SIM_REPO")
        or repo
        or Path(__file__).resolve().parents[2]
    )
    git_common = _absolute(common_dir) if common_dir else _git_common_dir(checkout)
    default_common_repo = git_common.parent if git_common.name == ".git" else checkout
    common_repo = _absolute(values.get("SIM_CANONICAL_REPO") or default_common_repo)
    projects_root = common_repo.parent
    rag_root = _absolute(values.get("SIM_RAG_ROOT") or projects_root / "rag_index")
    catalog = _absolute(
        values.get("SIM_CATALOG")
        or projects_root / "sim-catalog" / "references"
    )
    rag_python = _absolute(
        values.get("SIM_RAG_PYTHON")
        or common_repo / ".venv-rag" / "bin" / "python"
    )
    engine_python = _absolute(
        values.get("SIM_ENGINE_PYTHON")
        or common_repo / ".venv" / "bin" / "python"
    )
    return RagPaths(
        checkout,
        common_repo,
        projects_root,
        rag_root,
        catalog,
        engine_python,
        rag_python,
    )


def choose_index(paths: RagPaths, corpus: str) -> Path:
    """Use the full canonical index, with a narrow findings-only fallback."""
    if paths.full_index.is_dir():
        return paths.full_index
    if corpus in {"all", "finding"} and paths.findings_index.is_dir():
        return paths.findings_index
    raise FileNotFoundError(
        f"canonical full RAG index is unavailable at {paths.full_index}; "
        "set SIM_RAG_ROOT explicitly or rebuild with tools/rag/update_indexes.py --rebuild"
    )


def stable_document_id(
    source_type: str,
    path: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    catalog: str | os.PathLike[str],
) -> str:
    """Return a worktree-independent ID within the declared corpus root."""
    root = _absolute(catalog if source_type in {"catalog", "kandel", "paper"} else repo)
    source = _absolute(path)
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"RAG source escaped its declared root: {source}") from exc
    namespace = "catalog" if source_type in {"catalog", "kandel", "paper"} else "sim"
    return f"{namespace}:{relative.as_posix()}"
