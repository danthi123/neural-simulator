"""tools/agent_merge_policy.py: which changes the driver agent may merge into main by itself (owner ruling 2026-10-01,
"Routine merges"). The policy is a security boundary, so these tests pin both directions: routine work passes, and
every reserved class (sim code, gates, tools, preregs, HANDOFFs, edits to existing runners / tests / findings,
deletions, renames) is refused."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("agent_merge_policy", os.path.join(_ROOT, "tools", "agent_merge_policy.py"))
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


def _reserved(status, path, text="PROVISIONAL finding"):
    return policy.reason_reserved(status, path, lambda p: text)


def test_routine_changes_pass():
    for status, path in [
        ("A", "research/findings/raw/_ldc_outcome_timing/credit_immediate_onset30_w30_s8.json"),
        ("A", "research/findings/raw/_ldc_outcome_timing/credit_immediate_onset30_w30_s8.json.prov.json"),
        ("M", "docs/drafts/long-delay-credit/SPIKING-WORKLOG.md"),
        ("A", "docs/drafts/ecker-ou-field/AFFECTED.md"),
        ("A", "tests/test_new_scorer_helper.py"),
        ("A", "research/findings/2026-10-02-ldc-outcome-timing-pilot.md"),
        ("A", "research/runners/_ldc_outcome_timing_pilot.py"),
    ]:
        assert _reserved(status, path) is None, (status, path)


def test_reserved_paths_are_refused():
    for status, path in [
        ("M", "sim/config.py"),
        ("A", "sim/new_module.py"),
        ("M", "tools/gates/cfg_phantom_field.py"),
        ("A", "tools/gates/new_gate.py"),
        ("M", "tools/githooks/pre-commit"),
        ("M", "tools/agent_merge_policy.py"),
        ("M", "tools/agent_merge.sh"),
        ("M", "tools/iw2_score.py"),
        ("M", "research/runners/long_delay_credit_bg.py"),
        ("A", "research/runners/long_delay_credit_bg_v2.py"),
        ("M", "research/runners/_ldc_outcome_timing_pilot.py"),
        ("M", "docs/drafts/long-delay-credit/HANDOFF.md"),
        ("A", "docs/drafts/new-packet/HANDOFF.md"),
        ("A", "research/findings/2026-10-02-stage1-exit-PREREGISTRATION.md"),
        ("M", "docs/drafts/x/prereg-amendment-PREREG.md"),
        ("M", "research/coordination/LOCAL_LLM_RUNBOOK.md"),
        ("M", "tests/test_bounded_conductance_integration.py"),
        ("M", "research/findings/2026-09-27-tone-bg-nchannel-tonic-firing-wall.md"),
        ("A", "docs/plans/2026-10-02-new-plan.md"),
        ("M", "CLAUDE.md"),
        ("A", ".claude/skills/new/SKILL.md"),
        ("M", "requirements.txt"),
        ("M", "research/findings/raw/_ldc_outcome_timing/credit_immediate_onset30_w30_s8.json"),
        ("A", "research/findings/raw/_provenance/runs.jsonl"),
        ("M", "research/coordination/backlog_notes.md"),
        ("A", "research/coordination/lane_builds.jsonl"),
        ("A", "docs/drafts/x/scorer.py"),
        ("A", "tests/helpers/conftest_extra.py"),
    ]:
        assert _reserved(status, path) is not None, (status, path)


def test_deletions_renames_and_type_changes_are_refused_even_on_routine_paths():
    for status in ("D", "R", "C", "T"):
        assert _reserved(status, "research/findings/raw/_x/a.json") is not None
        assert _reserved(status, "docs/drafts/x/SPIKING-WORKLOG.md") is not None


def test_a_new_finding_must_say_provisional():
    assert _reserved("A", "research/findings/2026-10-02-x.md", text="GO 6/6, ratified") is not None
    assert _reserved("A", "research/findings/2026-10-02-x.md", text="Status: PROVISIONAL until Claude confirms") is None


def test_reserved_changes_lists_every_offender():
    bad = policy.reserved_changes(
        [("A", "research/findings/raw/a.json"), ("M", "sim/bridge.py"), ("D", "research/findings/raw/b.json")],
        lambda p: "")
    assert [p for p, _ in bad] == ["sim/bridge.py", "research/findings/raw/b.json"]


def test_agent_merge_sh_end_to_end(tmp_path):
    """agent_merge.sh in a scratch repo: a routine branch merges with --no-ff; a reserved branch is refused and main
    is left exactly as it was."""
    import shutil
    import subprocess

    def git(*a, check=True):
        return subprocess.run(["git", *a], cwd=tmp_path, check=check, capture_output=True, text=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "tools").mkdir()
    for f in ("agent_merge.sh", "agent_merge_policy.py"):
        shutil.copy(os.path.join(_ROOT, "tools", f), tmp_path / "tools" / f)
    (tmp_path / "sim").mkdir(); (tmp_path / "sim" / "core.py").write_text("x = 1\n")
    git("add", "-A"); git("commit", "-q", "-m", "base")

    git("checkout", "-q", "-b", "research/routine")
    raw = tmp_path / "research" / "findings" / "raw" / "_t"; raw.mkdir(parents=True)
    (raw / "a.json").write_text("{}\n")
    git("add", "-A"); git("commit", "-q", "-m", "data")
    git("checkout", "-q", "-b", "research/reserved", "main")
    (tmp_path / "sim" / "core.py").write_text("x = 2\n")
    git("add", "-A"); git("commit", "-q", "-m", "code")
    git("checkout", "-q", "main")

    run = lambda b: subprocess.run(["bash", "tools/agent_merge.sh", b], cwd=tmp_path, capture_output=True, text=True)
    before = git("rev-parse", "HEAD").stdout
    r = run("research/reserved")
    assert r.returncode == 2 and "sim/core.py" in r.stdout + r.stderr
    assert git("rev-parse", "HEAD").stdout == before and git("status", "--porcelain").stdout == ""

    r = run("research/routine")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(git("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()) == 3   # a real merge commit
    assert (tmp_path / "research/findings/raw/_t/a.json").exists()

    git("worktree", "add", "-q", str(tmp_path / "wt"), "research/reserved")
    r = subprocess.run(["bash", str(tmp_path / "tools/agent_merge.sh"), "research/routine"], cwd=tmp_path / "wt",
                       capture_output=True, text=True)
    assert r.returncode == 2 and "worktree" in r.stderr


def test_symlinks_and_submodules_are_refused_and_raw_z_parses_odd_paths():
    assert policy.reason_reserved("A", "docs/drafts/x/NOTES.md", lambda p: "", mode="120000") is not None
    assert policy.reason_reserved("A", "research/findings/raw/_x/sim", lambda p: "", mode="160000") is not None
    raw = ":000000 100644 0000000 abc1234 A\0research/findings/raw/a b\tc.json\0:100644 120000 abc1234 def5678 T\0docs/x.md\0"
    assert policy.parse_raw_z(raw) == [("A", "research/findings/raw/a b\tc.json", "100644"), ("T", "docs/x.md", "120000")]


def test_staged_backstop_allows_coordination_but_not_code():
    assert policy.reason_reserved("M", "research/coordination/LOCAL_LLM_RUNBOOK.md", lambda p: "", staged=True) is None
    assert policy.reason_reserved("M", "research/coordination/LOCAL_LLM_RUNBOOK.md", lambda p: "") is not None
    assert policy.reason_reserved("M", "sim/core.py", lambda p: "", staged=True) is not None


def test_draft_data_files_and_append_only_ledger_are_routine():
    assert _reserved("A", "docs/drafts/x/grouped-first-error-lines.txt") is None
    assert _reserved("M", "docs/drafts/x/cells.csv") is None
    assert _reserved("A", "docs/drafts/x/run.sh") is not None
    led = "research/findings/raw/_provenance/runs.jsonl"
    grow = lambda p: '{"a":1}\n{"b":2}\n'
    assert policy.reason_reserved("M", led, grow, old=lambda p: '{"a":1}\n') is None
    assert policy.reason_reserved("M", led, grow, old=lambda p: '{"z":0}\n') is not None   # a rewrite
    assert policy.reason_reserved("M", led, grow) is not None                                 # unprovable
    assert policy.reason_reserved("A", "research/findings/raw/_provenance/new.jsonl", grow, old=lambda p: "") is not None


def test_board_sync_stamp_is_append_only():
    grow = lambda p: "## a\n## b\n"
    assert policy.reason_reserved("M", "docs/.vikunja_sync", grow, old=lambda p: "## a\n") is None
    assert policy.reason_reserved("M", "docs/.vikunja_sync", grow, staged=True, old=lambda p: "## a\n") is None
    assert policy.reason_reserved("M", "docs/.vikunja_sync", grow, old=lambda p: "## z\n") is not None
    assert policy.reason_reserved("M", "docs/.vikunja_sync", grow) is not None


def test_failure_log_is_append_only_by_whole_rows():
    """research/FAILURE_LOG.md (Claude ruling 2026-10-04): growth by complete rows at the end is routine, on a branch
    merge and in a direct commit on main; every other change to it stays Claude's."""
    log = "research/FAILURE_LOG.md"
    old = "# Failure log\n\n| date | failure | gate |\n|---|---|---|\n| 2026-10-01 | a thing broke | `coverage` |\n"
    row = "| 2026-10-04 | T24 queued cells with no corpus check | `corpus_check_required` (caught at commit) |\n"
    ok = lambda new: policy.reason_reserved("M", log, lambda p: new, old=lambda p: old)
    assert ok(old + row) is None
    assert ok(old + row + row.replace("T24", "T25")) is None
    assert policy.reason_reserved("M", log, lambda p: old + row, staged=True, old=lambda p: old) is None
    # refused: an edit to an existing row, an inserted row, prose, a blank line, a heading, a partial row, a glued row
    assert ok(old.replace("a thing broke", "nothing broke") + row) is not None
    # a merge of an older branch inserts its rows mid-table: routine, as long as every old line survives in order
    assert ok(old.replace("|---|---|---|\n", "|---|---|---|\n" + row)) is None
    assert ok(old.replace("| date |", row + "| date |")) is not None                   # above the header
    assert ok(old.replace("| 2026-10-01 | a thing broke | `coverage` |\n", "") + row) is not None   # a deletion
    assert ok(old + "Some prose about the failure.\n") is not None
    assert ok(old + "\n" + row) is not None
    assert ok(old + "## New section\n") is not None
    assert ok(old + "| 2026-10-04 | only two cells |\n") is not None
    assert ok(old + "| 2026-10-04 | no newline at end | `coverage` |") is not None
    assert ok(old + row + "| 2026-10-04 | trailing | `coverage` | extra\n") is not None
    assert ok(old + "| 10/04/2026 | bad date | `coverage` |\n") is not None
    assert ok(old) is not None                                                         # nothing appended
    glued = old.rstrip("\n")
    assert policy.reason_reserved("M", log, lambda p: glued + row, old=lambda p: glued) is not None
    # unprovable growth, a new log, a deletion: reserved
    assert policy.reason_reserved("M", log, lambda p: old + row) is not None
    assert policy.reason_reserved("A", log, lambda p: row, old=lambda p: "") is not None
    assert policy.reason_reserved("D", log, lambda p: "", old=lambda p: old) is not None


def test_board_inserts_stay_reserved_until_the_owner_switches_them_on(monkeypatch):
    """GAP_CLOSURE_MISSION.md board lines: reserved while OWNER_APPROVED_BOARD_INSERTS is False (as shipped); when the
    owner turns it on, inserted lines that claim no PASS/GO are routine and every other change stays reserved."""
    board = "GAP_CLOSURE_MISSION.md"
    old = "# Board\n\n## STATE\n\n- line one\n- line two\n"
    nogo = old.replace("- line one\n", "- line one\n- **2026-10-02:** BG live-seam gate NO-GO (G4 only).\n")
    check = lambda new, **kw: policy.reason_reserved("M", board, lambda p: new, old=lambda p: old, **kw)
    monkeypatch.setattr(policy, "OWNER_APPROVED_BOARD_INSERTS", False)
    assert check(nogo) is not None
    monkeypatch.setattr(policy, "OWNER_APPROVED_BOARD_INSERTS", True)
    assert check(nogo) is None
    assert check(nogo, staged=True) is None
    assert check(old + "- T7 UNDEFINED(dev): instrument; FAIL 5/6 on cupy.\n") is None
    assert check(old + "- stage-1 C2 PASS 6/6\n") is not None
    assert check(old + "- the flip is a GO on six seeds\n") is not None
    assert check(old.replace("- line two\n", "- line 2\n")) is not None                 # an edit
    assert check(old.replace("- line two\n", "")) is not None                           # a deletion
    assert check(old) is not None                                                       # nothing inserted
    assert policy.reason_reserved("M", board, lambda p: nogo) is not None               # unprovable
