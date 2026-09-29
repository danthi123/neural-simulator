"""Tests for tools/inspection_digest.py (docs/plans/2026-09-29-local-model-transition-PLAN.md section 6).

Two groups:
1. Pure-parsing functions (findings frontmatter, NEEDS-CLAUDE items, FAILURE_LOG rows, pool/GPU log summaries):
   tested against synthetic fixture text/files via monkeypatched module-level paths -- no git involved.
2. Git-backed functions (branch listing, per-branch commits, new-finding-file discovery, --mark): tested against
   a small SCRATCH git repo built in tmp_path (never the real project repo), with `inspection_digest.ROOT`
   monkeypatched to point at it, so these are fast and independent of this repo's real (very large) history.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspection_digest as digest  # noqa: E402


# --------------------------------------------------------------------------------------- pure-parsing: findings

def test_summarize_finding_reads_frontmatter_and_filename_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "ROOT", tmp_path)
    rel = "research/findings/2026-09-25-something-NO-GO.md"
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\n"
        "type: finding\n"
        "status: no-go\n"
        "claim_check: measured\n"
        "prereg: research/findings/2026-09-24-something-PREREGISTRATION.md\n"
        "---\n"
        "ordinary body text, no special marker here\n"
    )
    info = digest.summarize_finding(rel)
    assert info["status"] == "no-go"
    assert info["claim_check"] == "measured"
    assert "PREREGISTRATION" in info["prereg"]
    assert "NO-GO" in info["filename_tags"]
    assert info["provisional"] is False


def test_summarize_finding_detects_provisional_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "ROOT", tmp_path)
    rel = "research/findings/2026-09-29-x-GO.md"
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_text("---\nstatus: go\n---\nThis verdict is PROVISIONAL until Claude re-derives it.\n")
    info = digest.summarize_finding(rel)
    assert info["provisional"] is True


# ---------------------------------------------------------------------------------- pure-parsing: NEEDS CLAUDE

RUNBOOK_FIXTURE = """# Runbook

## Some other section

- not a needs-claude item

## NEEDS CLAUDE

- 2026-09-20 10:00 (Claude) an old item, before the window
- 2026-09-27 17:20 (Claude) a recent item, inside the window
- 2026-09-28 09:00 (Claude) another recent item

## A later section

- 2026-09-29 should not be picked up, wrong section
"""


def test_needs_claude_items_filters_by_floor_date(tmp_path, monkeypatch):
    p = tmp_path / "RUNBOOK.md"
    p.write_text(RUNBOOK_FIXTURE)
    monkeypatch.setattr(digest, "RUNBOOK", p)
    items = digest.needs_claude_items("2026-09-25")
    assert len(items) == 2
    assert all("2026-09-2" in it and it.split()[1] >= "2026-09-25" for it in items)
    assert not any("2026-09-20" in it for it in items)
    assert not any("wrong section" in it for it in items)


def test_needs_claude_items_missing_runbook_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "RUNBOOK", tmp_path / "does-not-exist.md")
    assert digest.needs_claude_items("2020-01-01") == []


def test_render_needs_claude_section_does_not_double_bullet(tmp_path, monkeypatch):
    p = tmp_path / "RUNBOOK.md"
    p.write_text(RUNBOOK_FIXTURE)
    monkeypatch.setattr(digest, "RUNBOOK", p)
    section = digest.render_needs_claude_section("2026-09-25")
    for line in section.splitlines():
        assert not line.startswith("- -"), f"doubled bullet: {line!r}"


# ------------------------------------------------------------------------------------ pure-parsing: FAILURE_LOG

FAILURE_LOG_FIXTURE = """# Failure log

| date | failure | gate |
|---|---|---|
| 2026-09-20 | an old failure, before the window | gates/old_gate |
| 2026-09-26 | a recent failure with a | pipe in the text | gates/recent_gate |
| 2026-09-27 | NOT-GATEABLE case | NOT-GATEABLE: reason here |
"""


def test_failure_log_rows_filters_by_date_and_parses_columns(tmp_path, monkeypatch):
    p = tmp_path / "FAILURE_LOG.md"
    p.write_text(FAILURE_LOG_FIXTURE)
    monkeypatch.setattr(digest, "FAILURE_LOG", p)
    rows = digest.failure_log_rows("2026-09-25")
    dates = [r[0] for r in rows]
    assert dates == ["2026-09-26", "2026-09-27"]
    # the pipe-in-the-text row: gate column (last) must still be "gates/recent_gate", not swallow the pipe
    recent = next(r for r in rows if r[0] == "2026-09-26")
    assert recent[2] == "gates/recent_gate"
    assert "pipe in the text" in recent[1]


def test_failure_log_rows_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "FAILURE_LOG", tmp_path / "nope.md")
    assert digest.failure_log_rows("2020-01-01") == []


# -------------------------------------------------------------------------- pure-parsing: pool / gpu log summary

def test_dispatch_log_summary_counts_dispatch_and_refusals(tmp_path, monkeypatch):
    p = tmp_path / "dispatch.log"
    p.write_text(
        "[pool-dispatch] started 09:00:00 | queue=x | poll=60s | nodes=pool1 pool2\n"
        "[pool-dispatch] 09:00:05 pool1 <- SIM_BACKEND=numpy .venv/bin/python -m research.runners.x\n"
        "[pool-dispatch] 09:00:10 pool2 <- SIM_BACKEND=numpy .venv/bin/python -m research.runners.y\n"
        "[pool-dispatch] REFUSING to start: already running as pid 123\n"
        "[pool-dispatch] 09:00:15 pool1 <- SIM_BACKEND=numpy .venv/bin/python -m research.runners.z\n"
    )
    monkeypatch.setattr(digest, "DISPATCH_LOG", p)
    summary = digest.dispatch_log_summary()
    assert summary["dispatched"] == 3
    assert summary["refused"] == 1
    assert summary["dispatcher_restarts"] == 1
    assert summary["per_node"] == {"pool1": 2, "pool2": 1}


def test_dispatch_log_summary_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "DISPATCH_LOG", tmp_path / "nope.log")
    assert digest.dispatch_log_summary() is None


def test_gpu_log_summary_filters_by_date_and_counts_rc(tmp_path, monkeypatch):
    p = tmp_path / "gpu_queue.log"
    p.write_text(
        "some SIM_BRIDGE noise line, no timestamp shape\n"
        "2026-09-20 08:00:00 START: cd x && python -m y\n"
        "2026-09-20 08:05:00 DONE(rc=0): cd x && python -m y\n"
        "2026-09-27 10:00:00 START: cd x && python -m z\n"
        "2026-09-27 10:05:00 DONE(rc=1): cd x && python -m z\n"
        "2026-09-28 11:00:00 START: cd x && python -m w\n"
        "2026-09-28 11:05:00 DONE(rc=0): cd x && python -m w\n"
    )
    monkeypatch.setattr(digest, "GPU_LOG", p)
    summary = digest.gpu_log_summary("2026-09-25")
    assert summary["starts"] == 2
    assert summary["done_ok"] == 1
    assert summary["done_fail"] == 1


def test_gpu_log_summary_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "GPU_LOG", tmp_path / "nope.log")
    assert digest.gpu_log_summary("2020-01-01") is None


# ------------------------------------------------------------------------------------------- git helpers (misc)

def test_since_date_uses_marker_sha_commit_date(monkeypatch):
    monkeypatch.setattr(digest, "sha_commit_date", lambda sha: "2026-09-20")
    assert digest.since_date("deadbeef", True) == "2026-09-20"


def test_since_date_passthrough_for_a_plain_date():
    assert digest.since_date("2026-09-25", False) == "2026-09-25"


def test_since_date_unrecognized_form_does_not_crash():
    assert digest.since_date("7 days ago", False) == "1970-01-01"


# --------------------------------------------------------------------------------- git-backed: scratch repo

def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _init_scratch_repo(tmp_path):
    repo = tmp_path / "scratch_repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    (repo / "README.md").write_text("main v1\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "main: initial commit")

    (repo / "research" / "findings").mkdir(parents=True)
    (repo / "research" / "findings" / "2026-09-25-old-finding.md").write_text("---\nstatus: live\n---\nold\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main: add an old finding")

    _git(repo, "branch", "research/draft-x")
    _git(repo, "checkout", "-q", "research/draft-x")
    (repo / "research" / "findings" / "2026-09-29-new-finding-GO.md").write_text("---\nstatus: go\n---\nnew\n")
    (repo / "draft.txt").write_text("draft work\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "draft: add new work and a finding")
    _git(repo, "checkout", "-q", "main")

    return repo


def test_list_target_branches_groups_correctly(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr(digest, "ROOT", repo)
    main_b, draft_b, other_b = digest.list_target_branches()
    assert main_b == ["main"]
    assert draft_b == ["research/draft-x"]
    assert other_b == []


def test_branch_commits_excludes_what_main_already_has(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr(digest, "ROOT", repo)
    rc, out = digest.git("rev-list", "--max-parents=0", "main")
    root_sha = out.strip()

    main_commits = digest.branch_commits("main", root_sha, True, "1970-01-01")
    draft_commits = digest.branch_commits("research/draft-x", root_sha, True, "1970-01-01")

    assert any("add an old finding" in c for c in main_commits)
    # the draft branch must show ONLY its own new commit, not main's two commits it was cut from
    assert len(draft_commits) == 1
    assert "add new work and a finding" in draft_commits[0]


def test_new_finding_files_finds_additions_on_any_target_branch(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr(digest, "ROOT", repo)
    paths = digest.new_finding_files("1970-01-01")
    assert "research/findings/2026-09-25-old-finding.md" in paths
    assert "research/findings/2026-09-29-new-finding-GO.md" in paths


def test_gate_failure_commits_matches_gate_and_fail_cooccurrence(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr(digest, "ROOT", repo)
    (repo / "x.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fix(gates): a gate blocked the commit until fixed")
    (repo / "y.txt").write_text("y\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "docs: unrelated commit mentioning neither word")
    rc, out = digest.git("rev-list", "--max-parents=0", "main")
    root_sha = out.strip()
    commits = digest.gate_failure_commits(root_sha, True, "1970-01-01")
    assert any("gate blocked the commit" in c for c in commits)
    assert not any("unrelated commit" in c for c in commits)


def test_mark_writes_main_head_sha(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr(digest, "ROOT", repo)
    marker = tmp_path / "last_inspection.txt"
    monkeypatch.setattr(digest, "MARKER_FILE", marker)
    rc = digest.mark()
    assert rc == 0
    written = marker.read_text().strip()
    rc2, expected = digest.git("rev-parse", "main")
    assert written == expected.strip()


def test_resolve_since_prefers_marker_file_over_lookback(tmp_path, monkeypatch):
    marker = tmp_path / "last_inspection.txt"
    marker.write_text("abc1234\n")
    monkeypatch.setattr(digest, "MARKER_FILE", marker)
    since, label, is_sha = digest.resolve_since(None)
    assert since == "abc1234"
    assert is_sha is True


def test_resolve_since_falls_back_to_a_week_when_no_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "MARKER_FILE", tmp_path / "does-not-exist.txt")
    since, label, is_sha = digest.resolve_since(None)
    assert is_sha is False
    assert "days ago" in label


def test_resolve_since_explicit_since_wins(tmp_path, monkeypatch):
    marker = tmp_path / "last_inspection.txt"
    marker.write_text("abc1234\n")
    monkeypatch.setattr(digest, "MARKER_FILE", marker)
    since, label, is_sha = digest.resolve_since("2026-09-01")
    assert since == "2026-09-01"
    assert is_sha is False


# ----------------------------------------------------------------------------------------------- end-to-end size

def test_build_digest_end_to_end_on_scratch_repo_produces_all_sections(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr(digest, "ROOT", repo)
    monkeypatch.setattr(digest, "RUNBOOK", repo / "no-runbook.md")
    monkeypatch.setattr(digest, "FAILURE_LOG", repo / "no-failure-log.md")
    monkeypatch.setattr(digest, "DISPATCH_LOG", repo / "no-dispatch.log")
    monkeypatch.setattr(digest, "GPU_LOG", repo / "no-gpu.log")

    text = digest.build_digest("1970-01-01")
    for heading in ("## 1. Commits per branch", "## 2. New findings", "## 3. NEEDS-CLAUDE",
                    "## 4. New research/FAILURE_LOG.md rows", "## 5. Pool / GPU utilization",
                    "## 6. Commits noting a gate failure"):
        assert heading in text
    assert "2026-09-29-new-finding-GO.md" in text
    approx_tokens = len(text) // 4
    assert approx_tokens < 15000
