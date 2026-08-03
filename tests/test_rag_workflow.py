from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from tools.rag import check_workflow
from tools.rag.rag_paths import (
    RagPaths,
    choose_index,
    resolve_paths,
    stable_document_id,
)


def _layout(tmp_path: Path) -> RagPaths:
    common = tmp_path / "projects" / "sim"
    worktree = tmp_path / "projects" / "sim-worktrees" / "topic"
    return RagPaths(
        repo=worktree,
        common_repo=common,
        projects_root=common.parent,
        rag_root=common.parent / "rag_index",
        catalog=common.parent / "sim-catalog" / "references",
        engine_python=common / ".venv" / "bin" / "python",
        rag_python=common / ".venv-rag" / "bin" / "python",
    )


def test_linked_worktree_resolves_shared_index_catalog_and_python(tmp_path):
    worktree = tmp_path / "projects" / "sim-worktrees" / "topic"
    common_dir = tmp_path / "projects" / "sim" / ".git"
    paths = resolve_paths(worktree, env={}, common_dir=common_dir)
    assert paths.repo == worktree.resolve()
    assert paths.common_repo == (tmp_path / "projects" / "sim").resolve()
    assert paths.rag_root == (tmp_path / "projects" / "rag_index").resolve()
    assert paths.catalog == (
        tmp_path / "projects" / "sim-catalog" / "references"
    ).resolve()
    assert paths.rag_python == (
        tmp_path / "projects" / "sim" / ".venv-rag" / "bin" / "python"
    ).resolve()
    assert paths.engine_python == (
        tmp_path / "projects" / "sim" / ".venv" / "bin" / "python"
    ).resolve()


def test_explicit_overrides_do_not_fall_through_to_an_unrelated_index(tmp_path):
    worktree = tmp_path / "worktree"
    explicit = tmp_path / "explicit-index"
    paths = resolve_paths(
        worktree,
        env={"SIM_RAG_ROOT": str(explicit)},
        common_dir=tmp_path / "canonical" / ".git",
    )
    assert paths.rag_root == explicit.resolve()
    with pytest.raises(FileNotFoundError, match="canonical full RAG index"):
        choose_index(paths, "catalog")


def test_full_index_and_narrow_findings_fallback(tmp_path):
    paths = _layout(tmp_path)
    paths.findings_index.mkdir(parents=True)
    assert choose_index(paths, "finding") == paths.findings_index
    with pytest.raises(FileNotFoundError):
        choose_index(paths, "paper")
    paths.full_index.mkdir()
    assert choose_index(paths, "paper") == paths.full_index


def test_document_ids_are_stable_across_worktrees_and_reject_escape(tmp_path):
    first = tmp_path / "a" / "sim"
    second = tmp_path / "b" / "sim"
    catalog = tmp_path / "sim-catalog" / "references"
    rel = Path("research/findings/result.md")
    first_id = stable_document_id("finding", first / rel, first, catalog)
    second_id = stable_document_id("finding", second / rel, second, catalog)
    assert first_id == second_id == "sim:research/findings/result.md"
    assert stable_document_id(
        "paper", catalog / "textbooks/topic/paper.txt", first, catalog
    ) == "catalog:textbooks/topic/paper.txt"
    with pytest.raises(ValueError, match="escaped"):
        stable_document_id("finding", tmp_path / "outside.md", first, catalog)


def test_workflow_check_cannot_report_ready_for_inert_hooks_or_legacy_index(
    tmp_path, monkeypatch
):
    paths = _layout(tmp_path)
    hook_dir = paths.repo / "tools/githooks"
    hook_dir.mkdir(parents=True)
    for name in ("pre-commit", "post-commit"):
        (hook_dir / name).write_text("#!/bin/sh\n", encoding="utf-8")
    paths.rag_python.parent.mkdir(parents=True)
    paths.rag_python.write_text("", encoding="utf-8")
    paths.engine_python.parent.mkdir(parents=True)
    paths.engine_python.write_text("", encoding="utf-8")
    paths.full_index.mkdir(parents=True)
    paths.catalog.mkdir(parents=True)
    monkeypatch.setattr(check_workflow, "resolve_paths", lambda repo: paths)
    monkeypatch.setattr(
        check_workflow,
        "_git",
        lambda repo, *args: check_workflow.EXPECTED_HOOKS_PATH,
    )
    checks = {name: ok for name, ok, _ in check_workflow.workflow_status(paths.repo)}
    assert checks["pre-commit-executable"] is False
    assert checks["post-commit-executable"] is False
    assert checks["index-schema"] is False

    for path in (
        hook_dir / "pre-commit",
        hook_dir / "post-commit",
        paths.engine_python,
        paths.rag_python,
    ):
        path.chmod(path.stat().st_mode | 0o111)
    (paths.rag_root / ".rag_schema.json").write_text(
        json.dumps({"document_id_schema": check_workflow.EXPECTED_SCHEMA}),
        encoding="utf-8",
    )
    assert all(ok for _, ok, _ in check_workflow.workflow_status(paths.repo))


def test_repo_post_commit_hook_logs_every_nonrefresh_path():
    hook = Path("tools/githooks/post-commit").read_text(encoding="utf-8")
    assert "BLOCKED: RAG interpreter missing" in hook
    assert "SKIP: branch=" in hook
    assert "BLOCKED: legacy/missing index schema" in hook
    assert 'SIM_REPO="$REPO"' in hook


def test_post_commit_runs_from_linked_worktree_without_recursive_commit(tmp_path):
    canonical = tmp_path / "sim"
    topic = tmp_path / "sim-worktrees" / "topic"
    rag_root = tmp_path / "rag_index"
    fake_python = tmp_path / "fake-rag-python"
    refresh_log = tmp_path / "refresh.log"

    subprocess.run(
        ["git", "init", "--initial-branch=main", str(canonical)], check=True,
        capture_output=True, text=True,
    )
    for key, value in (("user.name", "RAG Test"), ("user.email", "rag@example.invalid")):
        subprocess.run(["git", "config", key, value], cwd=canonical, check=True)
    (canonical / "docs").mkdir()
    (canonical / "docs/note.md").write_text("initial\n", encoding="utf-8")
    hook_dir = canonical / "tools/githooks"
    hook_dir.mkdir(parents=True)
    hook = hook_dir / "post-commit"
    hook.write_text(
        Path("tools/githooks/post-commit").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    hook.chmod(0o755)
    updater = canonical / "tools/rag/update_indexes.py"
    updater.parent.mkdir(parents=True)
    updater.write_text("raise SystemExit('fake interpreter should intercept this')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=canonical, check=True)
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "commit", "-m", "initial"],
        cwd=canonical, check=True, capture_output=True, text=True,
    )
    topic.parent.mkdir()
    subprocess.run(
        ["git", "worktree", "add", "-b", "topic", str(topic)],
        cwd=canonical, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "config", "core.hooksPath", "tools/githooks"],
        cwd=topic, check=True,
    )
    rag_root.mkdir()
    (rag_root / ".rag_schema.json").write_text(
        json.dumps({"document_id_schema": "repo-relative-v1"}), encoding="utf-8"
    )
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s|%s\\n' \"$SIM_REPO\" \"$*\" >> \"$RAG_TEST_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    (topic / "docs/note.md").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/note.md"], cwd=topic, check=True)
    env = {
        **os.environ,
        "SIM_RAG_ROOT": str(rag_root),
        "SIM_RAG_PYTHON": str(fake_python),
        "SIM_RAG_REFRESH_BRANCH": "topic",
        "SIM_RAG_REFRESH_DELAY": "0",
        "RAG_TEST_LOG": str(refresh_log),
    }
    committed = subprocess.run(
        ["git", "commit", "-m", "trigger refresh"], cwd=topic, env=env,
        check=True, capture_output=True, text=True,
    )
    assert "ignored because it's not set as executable" not in committed.stderr
    for _ in range(100):
        if refresh_log.exists():
            break
        time.sleep(0.02)
    line = refresh_log.read_text(encoding="utf-8").strip()
    source_repo, updater_arg = line.split("|", 1)
    assert Path(source_repo) == topic.resolve()
    assert Path(updater_arg) == (topic / "tools/rag/update_indexes.py").resolve()
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=topic, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert count == "2"
