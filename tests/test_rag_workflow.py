from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.rag import check_workflow
from tools.rag.rag_paths import (
    RagPaths,
    choose_index,
    resolve_paths,
    stable_document_id,
)
from tools.rag.retrieval import _source_intent, candidate_count
from tools.rag.rag_eval import score_one


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
    ).absolute()


def test_virtualenv_python_paths_are_not_resolved_to_system_interpreter(tmp_path):
    common = tmp_path / "projects" / "sim"
    worktree = tmp_path / "projects" / "sim-worktrees" / "topic"
    for environment in (".venv", ".venv-rag"):
        executable = common / environment / "bin" / "python"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.symlink_to("/usr/bin/python3")
    paths = resolve_paths(worktree, env={}, common_dir=common / ".git")
    assert paths.rag_python == common / ".venv-rag" / "bin" / "python"
    assert paths.engine_python == common / ".venv" / "bin" / "python"


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
    assert 'SIM_REPO="$repo"' in hook
    assert "nohup" in hook
    assert "</dev/null" in hook
    assert "START: branch=" in hook
    assert "EXIT: status=" in hook


def test_retrieval_keeps_a_broad_hybrid_rerank_window():
    assert candidate_count(1) == 30
    assert candidate_count(5) == 30
    assert candidate_count(10) == 60


def test_named_source_intent_preserves_compact_version_tokens():
    def hit(source):
        return SimpleNamespace(node=SimpleNamespace(
            metadata={"source": source}, ref_doc_id=source
        ))

    query = "did neural vocal credit Gate B v4 pass or was it retired?"
    assert _source_intent(query, hit("neural-vocal-credit-gateB-v4-smoke-NO-GO.md")) \
        > _source_intent(query, hit("neural-vocal-credit-gateB-v5-smoke-QUALIFIED.md"))


def test_scientific_eval_requires_the_labeled_passage_not_just_the_source():
    hits = [
        {"source": "feature-catalog.md", "text": "SABI normally inhibits another interneuron."},
        {"source": "feature-catalog.md", "text": "FSI to MSN feedforward inhibition is powerful."},
    ]
    score = score_one(
        hits,
        ["feature-catalog"],
        5,
        must_contain=["FSI", "feedforward"],
        must_not_contain=["SABI normally inhibits"],
    )
    assert score["first_rel_rank"] == 2
    assert score["hit@1"] == 0
    assert score["hit@3"] == 1


def test_quality_evaluator_is_portable_and_fail_closed():
    evaluator = Path("tools/rag/rag_eval.py").read_text(encoding="utf-8")
    assert "E:\\Documents" not in evaluator
    assert 'RagRetriever(PATHS, corpus="all"' in evaluator
    assert "RAG_QUALITY_BLOCKED" in evaluator
    assert 'default=0.90' in evaluator
    launcher = Path("tools/rag/eval.sh")
    assert launcher.exists()


def test_index_refresh_runs_the_quality_floor():
    updater = Path("tools/rag/update_indexes.py").read_text(encoding="utf-8")
    check = updater.index("check_retrieval_quality(candidate_root)", updater.index("def main"))
    publish = updater.index("publish_candidate(candidate)", updater.index("def main"))
    manifest = updater.index('json.dump({"hash": indexed_hash', updater.index("def main"))
    assert check < publish < manifest
    assert '"--no-write"' in updater
    assert "SIM_RAG_SKIP_QUALITY" not in updater
    assert 'max(0.90, float(os.environ.get("SIM_RAG_MIN_MRR"' in updater
    assert "corpus changed during refresh; repeating" in updater


def test_post_commit_runs_from_linked_worktree_without_recursive_commit(tmp_path):
    canonical = tmp_path / "sim"
    topic = tmp_path / "sim-worktrees" / "topic"
    rag_root = tmp_path / "rag_index"
    fake_python = tmp_path / "fake-rag-python"
    refresh_log = tmp_path / "refresh.log"
    release = tmp_path / "release-refresh"

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
        "#!/bin/sh\n"
        "attempt=0\n"
        "while [ ! -e \"$RAG_TEST_RELEASE\" ] && [ \"$attempt\" -lt 500 ]; do\n"
        "  sleep 0.01\n"
        "  attempt=$((attempt + 1))\n"
        "done\n"
        "[ -e \"$RAG_TEST_RELEASE\" ] || exit 99\n"
        "printf '%s|%s\\n' \"$SIM_REPO\" \"$*\" >> \"$RAG_TEST_LOG\"\n",
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
        "RAG_TEST_RELEASE": str(release),
        "RAG_TEST_LOG": str(refresh_log),
    }
    committed = subprocess.run(
        ["git", "commit", "-m", "trigger refresh"], cwd=topic, env=env,
        check=True, capture_output=True, text=True,
    )
    assert "ignored because it's not set as executable" not in committed.stderr
    assert not refresh_log.exists(), "marker ran before the hook parent exited"
    release.write_text("parent exited\n", encoding="utf-8")
    for _ in range(150):
        if refresh_log.exists():
            break
        time.sleep(0.02)
    line = refresh_log.read_text(encoding="utf-8").strip()
    source_repo, updater_arg = line.split("|", 1)
    assert Path(source_repo) == topic.resolve()
    assert Path(updater_arg) == (topic / "tools/rag/update_indexes.py").resolve()
    autoupdate_log = (rag_root / "_autoupdate.log").read_text(encoding="utf-8")
    assert "[post-commit] START:" in autoupdate_log
    assert "[post-commit] EXIT: status=0" in autoupdate_log
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=topic, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert count == "2"
